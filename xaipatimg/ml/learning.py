from datetime import datetime
from os.path import join
import json
from pathlib import Path

import numpy as np
import pandas as pd
import os
import torch
import tqdm
import shutil
import wandb
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, confusion_matrix
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from xaipatimg.ml import resnet18_preprocess_no_norm
# from xaipatimg.ml.xai import _create_dirs, _get_subfolder


class PatImgDataset(torch.utils.data.Dataset):
    def __init__(self, db_dir, csv_dataset_filename, transform=None, target_transform=None, max_size=None):
        """
        Initialize the dataset object.
        :param db_dir: path to the root directory of the database
        :param csv_dataset_filename: name of the csv dataset file
        :param transform: pipeline for transforming input data
        :param target_transform: pipeline for transforming labels
        :param max_size: if not None, the maximum number of images to load
        """
        self.db_dir = db_dir
        dataset_csv = pd.read_csv(os.path.join(
            db_dir, "datasets", csv_dataset_filename))
        self.img_list = dataset_csv["path"]
        self.img_labels = dataset_csv["class"]
        self.transform = transform
        self.target_transform = target_transform
        self.images_cache = []

        if max_size is not None:
            self.img_list = self.img_list[:max_size]
            self.img_labels = self.img_labels[:max_size]

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
        image = self.images_cache[idx]
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

    def get_id(self, idx):
        """
        Returns the id of the image in the json database for a given index in the dataset.
        Assumes that the filename is the same as the ID.
        :param idx: index
        :return:
        """
        # The filename is the id
        return Path(self.img_list[idx]).stem


def compute_mean_std_dataset(db_dir, dataset_filename, preprocess_no_norm):
    """
    Compute the mean and std of the transformed dataset.
    :param db_dir: root of the db
    :param dataset_filename: name of the csv dataset file
    :param preprocess_no_norm: preprocessing pipeline without normalization
    :return: tuple ([mean on every channel], [std on every channel])
    """
    dataset_no_norm = PatImgDataset(
        db_dir, dataset_filename, transform=preprocess_no_norm)
    alldata_no_norm = np.array([dataset_no_norm[i][0]
                               for i in range(len(dataset_no_norm))])
    means = [np.mean(x).astype(float) for x in
             [alldata_no_norm[:, channel, :] for channel in range(alldata_no_norm.shape[1])]]
    stds = [np.std(x).astype(float) for x in
            [alldata_no_norm[:, channel, :] for channel in range(alldata_no_norm.shape[1])]]
    return means, stds

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
    checkpoint_path = join(model_dir, "final_model")

    # Stop if accuracy threshold is reached
    if vaccuracy >= target_accuracy:
        cap_path = join(model_dir, checkpoint_path)
        torch.save(model.state_dict(), cap_path)
        print(f"Accuracy cap hit at {label} {step}")
        return True, counter, best_loss

    if patience is not None:
        # Check for loss improvement
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
            torch.save(model.state_dict(), join(model_dir, checkpoint_path))
            with open(join(model_dir, "final_model_epoch"), "w") as f:
                f.write(str(step))
            return False, counter, best_loss
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at {label} {step}")
                return True, counter, best_loss
            return False, counter, best_loss
    return False, counter, best_loss


def train_resnet18_model(db_dir, train_dataset_filename, valid_dataset_filename, model_dir, device="cuda:0", training_epochs=90, lr=0.1,
                         momentum=0.9, weight_decay=1e-4, batch_size=32, lr_step_size=30, lr_gamma=0.1, train_loss_write_period_logs=100,
                         target_accuracy=1.0, training_mode="batch", patience=None, interval_batch=200):
    """
    Perform the training of the given model.
    The default hyper-parameters correspond to the ones that were used to train ResNet18 model. The stochastic
    gradient descent optimizer (SGD) is used along with the StepLR scheduler.
    https://github.com/pytorch/vision/tree/main/references/classification

    :param db_dir: path to the root directory of the database
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
    :return:
    """
    os.makedirs(model_dir, exist_ok=True)

    # --- Setup ---
    rule_name = Path(train_dataset_filename).stem.split('_')[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(project="xaipatimg", name=f"{rule_name}-{timestamp}", config={
        "learning_rate": lr,
        "epochs": training_epochs,
        "batch_size": batch_size,
        "lr_step_size": lr_step_size,
        "lr_gamma": lr_gamma,
        "target_accuracy": target_accuracy,
        "training_mode": training_mode,
        "patience": patience,
        "interval_batch": interval_batch
    })

    means, stds = compute_mean_std_dataset(
        db_dir, train_dataset_filename, resnet18_preprocess_no_norm)
    print(f"Train dataset statistics : " + str(means) + " " + str(stds))
    with open(os.path.join(model_dir, "train_set_stats.json"), "w") as f:
        json.dump({"mean": means, "std": stds}, f)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    dataset_train = PatImgDataset(db_dir, train_dataset_filename, transform=preprocess)
    dataset_valid = PatImgDataset(db_dir, valid_dataset_filename, transform=preprocess)
    training_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'resnet18', pretrained=False)
    model.fc = Linear(512, 2)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    writer = SummaryWriter(
        join(model_dir, "run/") + f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # --- Training State ---
    best_vloss = np.inf
    counter = 0
    global_step = 0
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

        wandb_log = {"training_loss": current_avg_tloss,
                     "validation_loss": avg_vloss, "vaccuracy": vaccuracy}
       
        if training_mode == "epoch":
            wandb_log["epoch"] = step
      
        else:
            wandb_log["global_step"] = step
        wandb.log(wandb_log)

        early_stop, new_counter, new_best_vloss = _check_early_stopping(
            vaccuracy, target_accuracy, avg_vloss, best_vloss, counter, patience, model, model_dir, training_mode, step)

        counter = new_counter
        best_vloss = new_best_vloss
        if early_stop:
            stop_training = True

    # --- Training Loop ---
    running_tloss = 0.0
    train_batches = 0

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
            train_batches += 1
            global_step += 1

            if global_step % train_loss_write_period_logs == 0:
                avg_loss = running_tloss / train_batches
                writer.add_scalar('Loss/train', avg_loss, global_step)

            # training_mode == batch
            if training_mode == "batch" and global_step % interval_batch == 0:
                avg_tloss = running_tloss / train_batches
                _validate_and_log(avg_tloss, global_step)
                running_tloss = 0.0
                train_batches = 0

        #   #training_mode == epoch mode 
        if training_mode == "epoch" and train_batches > 0 and not stop_training:
            avg_loss = running_tloss / train_batches
            _validate_and_log(avg_loss, epoch + 1)
            running_tloss = 0.0
            train_batches = 0

        # ── catch‑up validation if batch‑mode hasn’t fired this epoch ─
        if training_mode == "batch" and train_batches > 0 and not stop_training:
            avg_tloss = running_tloss / train_batches
            _validate_and_log(avg_tloss, global_step)
            running_tloss = 0.0
            train_batches = 0

        scheduler.step()

    writer.close()
    wandb.finish()

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


def load_resnet18_based_model(model_dir, device):
    """
    Creating a resnet18 model with two output features and loading the weights from the model stored in the given
    directory.
    :param model_dir: path to model directory (contains a file called final_model).
    :param device: device on which to run the model.
    :return: the model.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = Linear(in_features=512, out_features=2, bias=True)

    model.load_state_dict(torch.load(os.path.join(model_dir, "final_model"), weights_only=True))
    model.eval()
    return model.to(device)


def get_dataset_transformed(db_dir, model_dir, dataset_filename, max_size=None):
    """
    Return a PatImgDataset instance for the given dataset, with a transformation that includes normalization of the data
    according to the values observed on the training dataset of the given model.
    :param db_dir: path to the directory that contains the database.
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

    return PatImgDataset(db_dir, dataset_filename, transform=preprocess, max_size=max_size)


def compute_resnet18_model_scores(db_dir, train_dataset_filename, test_dataset_filename, valid_dataset_filename,
                                  model_dir, device="cuda:0"):
    """
    Computes and writes the scores of the given model into a file in the model directory. The scores are evaluated
    on the training, validation and test sets.
    :param db_dir: path to the directory of the database.
    :param train_dataset_filename: filename of the csv training dataset file.
    :param test_dataset_filename: filename of the csv test dataset file.
    :param valid_dataset_filename: filename of the csv validation dataset file.
    :param model_dir: path to the directory of the model.
    :param device: device where to perform the computations.
    :return:
    """

    # Importing the data
    dataset_train = get_dataset_transformed(db_dir, model_dir, train_dataset_filename)
    dataset_test = get_dataset_transformed(db_dir, model_dir, test_dataset_filename)
    dataset_valid = get_dataset_transformed(db_dir, model_dir, valid_dataset_filename)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=100, shuffle=False)

    # Loading the model
    model = load_resnet18_based_model(model_dir, device)

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


def save_classification(db_dir, test_dataset_filename, model_dir, classification_dir, device="cuda:0", max_items=None):
    from xaipatimg.ml.xai import _create_dirs, _get_subfolder
    """
    Copy every image listed in test dataset into TP / TN / FP / FN folders to observer what the model got right or wrong.
    :param db_dir: path to the root directory of the database.
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
    model = load_resnet18_based_model(model_dir, device)
    model.eval()

    dataset = get_dataset_transformed(
        db_dir=db_dir,
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
