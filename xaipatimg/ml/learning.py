import sys
from datetime import datetime
from os.path import join
import json

import numpy as np
import pandas as pd
import os
import torch
import tqdm
import shutil
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, confusion_matrix
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from xaipatimg.ml import resnet18_preprocess_no_norm


class PatImgDataset(torch.utils.data.Dataset):
    def __init__(self, db_dir, datasets_dir_path, csv_dataset_filename, transform=None, target_transform=None,
                 max_size=None, keep_images_in_cache=False):
        """
        Initialize the dataset object.
        :param db_dir: path to the root directory of the database
        :param csv_dataset_filename: name of the csv dataset file
        :param transform: pipeline for transforming input data
        :param target_transform: pipeline for transforming labels
        :param max_size: if not None, the maximum number of images to load
        :param keep_images_in_cache: if True, keeping all images in memory
        """
        self.db_dir = db_dir
        self.datasets_dir_path = datasets_dir_path
        dataset_csv = pd.read_csv(os.path.join(datasets_dir_path, csv_dataset_filename))
        self.img_list = dataset_csv["path"]
        self.img_labels = dataset_csv["class"]
        self.transform = transform
        self.target_transform = target_transform
        self.images_cache = []

        if max_size is not None:
            self.img_list = self.img_list[:max_size]
            self.img_labels = self.img_labels[:max_size]

        if keep_images_in_cache:
            print("Loading dataset content for " + csv_dataset_filename)
            for idx in tqdm.tqdm(range(len(self.img_list))):
                img_path = os.path.join(self.db_dir, self.img_list[idx])
                image = Image.open(img_path)
                if self.transform is not None:
                    image = self.transform(image)
                self.images_cache.append(image)

    def __len__(self):
        """
        Returns the size of the dataset.
        :return:
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Returns the transformed image for learning and its label for a given index in the dataset.
        :param idx: index
        :return: X, y
        """

        if self.images_cache:
            image = self.images_cache[idx]
        else:
            img_path = os.path.join(self.db_dir, self.img_list[idx])
            image = Image.open(img_path)
            if self.transform is not None:
                image = self.transform(image)

        label = self.img_labels[idx]
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_path(self, idx):
        """
        Returns the path of the image for a given index in the dataset.
        :param idx: index
        :return: path to the image
        """
        return os.path.join(self.db_dir, self.img_list[idx])

    def get_image_name(self, idx):
        """
        Returns the name of the image for a given index in the dataset.
        :param idx: index
        :return: name of the image
        """
        return self.img_list[idx]

    def get_id(self, idx):
        """
        Returns the id of the image in the json database for a given index in the dataset.
        Assumes that the filename is the same as the ID.
        :param idx: index
        :return:
        """
        # The filename is the id
        return Path(self.img_list[idx]).stem

    def get_label(self, idx):
        """
        Returns the label of the image with the given index.
        :param idx: index
        :return:
        """
        return self.img_labels[idx]


def compute_mean_std_dataset(db_dir, datasets_dir_path, dataset_filename, preprocess_no_norm):
    """
    Compute the mean and std of the transformed dataset.
    :param db_dir: root of the db
    :param datasets_dir_path: path to the datasets directory
    :param dataset_filename: name of the csv dataset file
    :param preprocess_no_norm: preprocessing pipeline without normalization
    :return: tuple ([mean on every channel], [std on every channel])
    """
    dataset_no_norm = PatImgDataset(
        db_dir, datasets_dir_path, dataset_filename, transform=preprocess_no_norm)

    loader = torch.utils.data.DataLoader(dataset_no_norm, batch_size=100, shuffle=False)

    count = 0
    mean = 0.0
    sqmean = 0.0

    for x, _ in tqdm.tqdm(loader):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        pixels = b * h * w
        count += pixels

        mean += x.sum(dim=[0, 2, 3])
        sqmean += (x ** 2).sum(dim=[0, 2, 3])

    mean /= count
    std = (sqmean / count - mean ** 2).sqrt()
    return mean.numpy().tolist(), std.numpy().tolist()

def _check_early_stopping(vaccuracy, target_accuracy, current_loss, best_loss, counter, patience, model, model_dir, mode, step):
    """
    Early stopping check on accuracy threshold and loss.
    :param vaccuracy: the current accuracy on validation data
    :param target_accuracy: the desired accuracy the model should reach
    :param current_loss: current validation loss
    :param best_loss: best lowest validation loss seen so far
    :param counter: count how many consecutive evaluations have not improved best_loss.
    :param patience: the number of times in a row that the validation loss does not improve before training is stopped early.
    :param model: save the model when the condition is met
    :param model_dir: directory to save model checkpoint
    :param mode: 'epoch' validate at the end of each epoch or 'batch' validate at every interval batches
    :param step: epoch number or global step
    :return: tuple (stop_training, new_counter, new_best_loss)
    """
    label = "Epoch" if mode == "epoch" else "Step"
    model_path = join(model_dir, "final_model")

    # Stop if accuracy threshold is reached
    if vaccuracy >= target_accuracy:
        cap_path = join(model_dir, model_path)
        torch.save(model.state_dict(), cap_path)
        print(f"Accuracy cap hit at {label} {step} : {vaccuracy} >= {target_accuracy}")
        return True, counter, best_loss
    elif target_accuracy <= 1.0:
        print(f"Accuracy cap NOT hit at {label} {step} : {vaccuracy} < {target_accuracy}")

    if patience is None:
        return False, counter, best_loss

    # Patience criterion if defined
    if current_loss < best_loss:
        best_loss = current_loss
        counter = 0
        torch.save(model.state_dict(), join(model_dir, model_path))
        with open(join(model_dir, "final_model_epoch"), "w") as f:
            f.write(str(step))
        return False, counter, best_loss
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at {label} {step}")
            return True, counter, best_loss
        return False, counter, best_loss


def train_resnet_model(db_dir, datasets_dir_path, train_dataset_filename, valid_dataset_filename,
                         model_dir, device="cuda:0", training_epochs=90, lr=0.1, momentum=0.9, weight_decay=1e-4,
                         batch_size=32, lr_step_size=30, lr_gamma=0.1, train_loss_write_period_logs=100,
                         target_accuracy=1.0, training_mode="batch", patience=None, interval_batch=200,
                         resnet_type="resnet18"):
    """
    Perform the training of the given model.
    The default hyper-parameters correspond to the ones that were used to train ResNet18 model. The stochastic
    gradient descent optimizer (SGD) is used along with the StepLR scheduler.
    https://github.com/pytorch/vision/tree/main/references/classification

    :param db_dir: path to the root directory of the database
    :param datasets_dir_path: path to the dictory where datasets are saved
    :param train_dataset_filename: filename of the csv training dataset file
    :param valid_dataset_filename: filename of the csv validation dataset file
    :param model_dir: directory to save model and tensorboard logs
    :param device: device where the computation is done (default : cuda:0)
    :param training_epochs: number of training epochs
    :param lr: learning rate
    :param momentum: momentum
    :param weight_decay: weight decay
    :param batch_size: batch size
    :param lr_step_size: learning rate step size (Step LR scheduler)
    :param lr_gamma: learning rate gamma (Step LR scheduler)
    :param train_loss_write_period_logs: period between two recordings of the training loss in the logs
    :param target_accuracy: stop the model from training once the desired accuracy on the validation dataset is reached
    :param training_mode: decides whether the model runs training and validation checks per 'batch' or per 'epoch'.
                         'epoch' for validation at the end of each epoch,
                         'batch' for validation at every interval batches.
    :param patience: the number of times in a row that the validation loss does not improve before training is stopped early.
    :param interval_batch: number of batches between each validation
    :param resnet_type: either "resnet18" or "resnet50"
    :return:
    """
    os.makedirs(model_dir, exist_ok=True)

    means, stds = compute_mean_std_dataset(
        db_dir, datasets_dir_path, train_dataset_filename, resnet18_preprocess_no_norm)
    print(f"Train dataset statistics : " + str(means) + " " + str(stds))
    with open(os.path.join(model_dir, "train_set_stats.json"), "w") as f:
        json.dump({"mean": means, "std": stds}, f)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    dataset_train = PatImgDataset(db_dir, datasets_dir_path, train_dataset_filename, transform=preprocess)
    dataset_valid = PatImgDataset(db_dir, datasets_dir_path, valid_dataset_filename, transform=preprocess,
                                  keep_images_in_cache=True)
    training_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    if os.path.isfile(os.path.join(model_dir, "final_model")):
        print("loading existing model")
        model = load_resnet_based_model(model_dir, device, resnet_type)

    else:
        if resnet_type == "resnet18":
            model = torch.hub.load('pytorch/vision:v0.10.0',
                                   'resnet18', pretrained=False)
            model.fc = Linear(512, 2)
            model = model.to(device)

        elif resnet_type == "resnet50":

            model = torch.hub.load('pytorch/vision:v0.10.0',
                                   'resnet50', pretrained=False)
            model.fc = Linear(2048, 2)
            model = model.to(device)

        else:
            raise ValueError(f"Unknown resnet type {resnet_type}")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    writer = SummaryWriter(
        join(model_dir, "run/") + f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # --- Training State ---
    best_vloss = np.inf
    counter = 0
    global_batch_nb = 0
    stop_training = False
    vaccuracies = []

    # --- Validation Helper ---
    def _validate_and_log(current_avg_tloss, step):
        nonlocal best_vloss, counter, stop_training, vaccuracies

        model.eval()
        running_vloss = 0.0
        correct = 0
        with torch.no_grad():
            for i, vdata in enumerate(valid_loader):
                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
                vouts = model(vinputs.float())
                running_vloss += loss_fn(vouts, vlabels).item()
                probs = torch.softmax(vouts, dim=1)
                correct += (probs.argmax(dim=1) == vlabels).sum().item()
        model.train()

        avg_vloss = running_vloss / (i + 1)
        vaccuracy = correct / len(dataset_valid)
        vaccuracies.append(vaccuracy)

        print(f'LOSS train {current_avg_tloss:.4f} valid {avg_vloss:.4f}')

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': current_avg_tloss,
                               'Validation': avg_vloss},
                           step)
        writer.add_scalars('Validation accuracy',
                           {'Validation': vaccuracy}, step)
        writer.flush()

        early_stop, new_counter, new_best_vloss = _check_early_stopping(
            vaccuracy, target_accuracy, avg_vloss, best_vloss, counter, patience, model, model_dir, training_mode, step)

        counter = new_counter
        best_vloss = new_best_vloss
        if early_stop:
            stop_training = True

    # --- Training Loop ---
    running_tloss = 0.0
    curr_epoch_batch_nb = 0

    for epoch in range(training_epochs):
        if stop_training:
            break

        print(f"EPOCH {epoch + 1}:")
        model.train()

        for batch_idx, data in enumerate(training_loader, 1):
            if stop_training:
                break

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs.float()), labels)
            loss.backward()
            optimizer.step()

            running_tloss += loss.item()
            curr_epoch_batch_nb += 1
            global_batch_nb += 1

            if global_batch_nb % train_loss_write_period_logs == 0:
                avg_loss = running_tloss / curr_epoch_batch_nb
                writer.add_scalar('Loss/train', avg_loss, global_batch_nb)

            # training_mode == batch
            if training_mode == "batch" and global_batch_nb % interval_batch == 0:
                avg_tloss = running_tloss / curr_epoch_batch_nb
                _validate_and_log(avg_tloss, global_batch_nb)
                running_tloss = 0.0
                curr_epoch_batch_nb = 0

        #   #training_mode == epoch mode 
        if training_mode == "epoch" and curr_epoch_batch_nb > 0 and not stop_training:
            avg_loss = running_tloss / curr_epoch_batch_nb
            _validate_and_log(avg_loss, epoch + 1)
            running_tloss = 0.0
            curr_epoch_batch_nb = 0

        # ── catch‑up validation if batch‑mode hasn’t fired this epoch ─
        if training_mode == "batch" and curr_epoch_batch_nb > 0 and not stop_training:
            avg_tloss = running_tloss / curr_epoch_batch_nb
            _validate_and_log(avg_tloss, global_batch_nb)
            running_tloss = 0.0
            curr_epoch_batch_nb = 0

        scheduler.step()

    writer.close()

    final_model_path = os.path.join(model_dir, "final_model")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete")


def make_prediction(model, X, device):
    """
    Getting the prediction for the given model and the given transformed input tensor.
    :param model: model.
    :param X: transformed input tensor.
    :param device: device where to perform the computation.
    :return:
    """
    with torch.no_grad():
        outputs = model(X.to(device).float())
        probabilities = torch.nn.functional.softmax(
            outputs, dim=1).cpu().numpy().reshape(-1, 2).T[1].tolist()
    y_pred = np.round(probabilities).astype(int)
    return y_pred, probabilities


def _compute_scores(data_loader, model, device):
    probabilities_all = []
    y_true_all = []
    y_pred_all = []
    for i, data in enumerate(data_loader):
        y_pred, probs = make_prediction(model, data[0], device)
        y_pred_all.extend(y_pred)
        probabilities_all.extend(probs)
        y_true_all.extend(data[1].numpy().reshape(-1,))

    tn, fp, fn, tp = confusion_matrix(y_pred_all, y_true_all).ravel()
    return {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "precision": precision_score(y_true_all, y_pred_all),
        "recall": recall_score(y_true_all, y_pred_all),
        "roc_auc": float(roc_auc_score(y_true_all, probabilities_all)),
        "confusion matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        }
    }

def load_resnet_based_model(model_dir, device, resnet_type="resnet18"):
    """
    Creating a resnet18 model with two output features and loading the weights from the model stored in the given
    directory.
    :param model_dir: path to model directory (contains a file called final_model).
    :param device: device on which to run the model.
    :param resnet_type: Either "resnet18" or "resnet50".
    :return: the model.
    """

    if resnet_type == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model.fc = Linear(in_features=512, out_features=2, bias=True)

    elif resnet_type == "resnet50":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        model.fc = Linear(in_features=2048, out_features=2, bias=True)

    else:
        raise ValueError(f"Unknown resnet type {resnet_type}")

    model.load_state_dict(torch.load(os.path.join(model_dir, "final_model"), weights_only=True, map_location=device))
    model.eval()
    return model.to(device)


def get_dataset_transformed(db_dir, datasets_dir_path, model_dir, dataset_filename, max_size=None):
    """
    Return a PatImgDataset instance for the given dataset, with a transformation that includes normalization of the data
    according to the values observed on the training dataset of the given model.
    :param db_dir: path to the directory that contains the database.
    :param datasets_dir_path: path to the directory that contains the datasets.
    :param model_dir: path to the directory that contains the model.
    :param dataset_filename: name of the dataset to load.
    :param max_size: maximum size of the dataset to load.
    :return: PatImgDataset instance
    """

    with open(os.path.join(model_dir, "train_set_stats.json"), "r") as f:
        stats_dict = json.load(f)

    means, stds = stats_dict["mean"], stats_dict["std"]

    # Definition of complete preprocessing pipeline that includes normalization based on the training data
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    return PatImgDataset(db_dir, datasets_dir_path, dataset_filename, transform=preprocess, max_size=max_size)


def compute_resnet_model_scores(db_dir, datasets_dir_path, train_dataset_filename, test_dataset_filename,
                                  valid_dataset_filename, model_dir, device="cuda:0", resnet_type="resnet18"):
    """
    Computes and writes the scores of the given model into a file in the model directory. The scores are evaluated
    on the training, validation and test sets.
    :param db_dir: path to the directory of the database.
    :param datasets_dir_path: path to the directory that contains the datasets.
    :param train_dataset_filename: filename of the csv training dataset file.
    :param test_dataset_filename: filename of the csv test dataset file.
    :param valid_dataset_filename: filename of the csv validation dataset file.
    :param model_dir: path to the directory of the model.
    :param device: device where to perform the computations.
    :param resnet_type: either "resnet18" or "resnet50".
    :return:
    """

    # Importing the data
    dataset_train = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, train_dataset_filename)
    dataset_test = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, test_dataset_filename)
    dataset_valid = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, valid_dataset_filename)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=100, shuffle=False)

    # Loading the model
    model = load_resnet_based_model(model_dir, device, resnet_type=resnet_type)

    # Computing the scores
    results = {
        "train": _compute_scores(train_loader, model, device),
        "test": _compute_scores(test_loader, model, device),
        "valid": _compute_scores(valid_loader, model, device)
    }

    # Writing the results
    with open(join(model_dir, "results.json"), "w") as f:
        json.dump(results, f)

    print(results)
    return results


def save_classification(db_dir, datasets_dir_path, test_dataset_filename, model_dir, classification_dir,
                        device="cuda:0", max_items=None, resnet_type="resnet18"):
    from xaipatimg.ml.xai import _create_dirs, _get_subfolder
    """
    Copy every image listed in test dataset into TP / TN / FP / FN folders to observer what the model got right or wrong.
    :param db_dir: path to the root directory of the database.
    :param datasets_dir_path: path to the directory that contains the datasets.
    :param test_dataset_filename: filename of the csv test dataset file.
    :param model_dir: path to the directory of the model.
    :param classification_dir: path to the root directory of the classified image
    :param max_items: maximum number of items to copy.
    :param device: device on which to run the model.
    """

    split_name = Path(test_dataset_filename).stem.rsplit("_", 1)[0]
    classification_dir = os.path.join(classification_dir, split_name)

    # makes TP / TN / FP / FN sub-folders
    _create_dirs(classification_dir)

    # Load model and dataset
    model = load_resnet_based_model(model_dir, device, resnet_type=resnet_type)
    model.eval()

    dataset = get_dataset_transformed(
        db_dir=db_dir,
        datasets_dir_path=datasets_dir_path,
        model_dir=model_dir,
        dataset_filename=test_dataset_filename,
        max_size=max_items,
    )

    # Run inference & copy originals
    for idx in range(len(dataset)):
        img, true_label = dataset[idx]

        pred_label, _ = make_prediction(model, img.unsqueeze(0), device)
        pred_label = int(pred_label[0])

        subfolder = _get_subfolder(pred_label, true_label)
        if subfolder is None:  # guard against unexpected labels
            raise ValueError(
                f"Unsupported label pair: pred={pred_label}, true={true_label}")

        source = dataset.get_path(idx)
        output = os.path.join(classification_dir,
                              subfolder, os.path.basename(source))
        shutil.copy2(source, output)
