from datetime import datetime
from os.path import join
import json
import numpy as np
import pandas as pd
import os
import torch
import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, confusion_matrix
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from xaipatimg.ml import resnet18_preprocess_no_norm


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


def _train_epoch(training_loader, model, optimizer, loss_fn, device, epoch_index, tb_writer, train_loss_write_period):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data[0].to(device), data[1].to(device)
        print("newbatch " + str(i))
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.float())

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if (i + 1) % train_loss_write_period == 0:
            last_loss = running_loss / train_loss_write_period
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train_resnet18_model(db_dir, train_dataset_filename, valid_dataset_filename, model_dir, device="cuda:0",
                         training_epochs=90, lr=0.1, momentum=0.9, weight_decay=1e-4, batch_size=32, lr_step_size=30,
                         lr_gamma=0.1, train_loss_write_period_logs=100, target_accuracy=0.8):
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
    :return:
    """
    os.makedirs(model_dir, exist_ok=True)

    # Computing training dataset means and stds
    means, stds = compute_mean_std_dataset(
        db_dir, train_dataset_filename, resnet18_preprocess_no_norm)
    print("Train dataset statistics : " + str(means) + " " + str(stds))

    # Writing training set statistics to file
    with open(os.path.join(model_dir, "train_set_stats.json"), "w") as f:
        json.dump({"mean": means, "std": stds}, f)

    # Definition of complete preprocessing pipeline that includes normalization based on the training data
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    # Importing the data
    dataset_train = PatImgDataset(
        db_dir, train_dataset_filename, transform=preprocess)
    dataset_valid = PatImgDataset(
        db_dir, valid_dataset_filename, transform=preprocess)
    training_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False)

    # Creating the model
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'resnet18', pretrained=False)
    model.fc = Linear(in_features=512, out_features=2, bias=True)
    model = model.to(device)

    # Creating the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Creating the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Creating the scheduler
    StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Initialization of the variables for the training
    epoch_number = 0
    vaccuracies = []
    best_vloss = np.inf
    patience = 10
    counter = 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(join(model_dir, "run/") + '_{}'.format(timestamp))

    # Main training loop
    for epoch in range(training_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = _train_epoch(training_loader, model, optimizer, loss_fn, device, epoch_number, writer,
                                train_loss_write_period_logs)

        running_vloss = 0.0

        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        correct = 0
        size = 0
        with torch.no_grad():
            for i, vdata in enumerate(valid_loader):
                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
                voutputs = model(vinputs.float())
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                probabilities = torch.nn.functional.softmax(voutputs, dim=1)
                correct += (probabilities.round().int().T[1] == vlabels).sum()
                size += len(vdata)

        vaccuracy = correct / len(dataset_valid)
        vaccuracies.append(vaccuracy.cpu())
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.add_scalars('Validation accuracy',
                           {'Validation': vaccuracy},
                           epoch_number + 1)
        writer.flush()

        # accuracy threshold
        if vaccuracy >= target_accuracy:
            cap_path = join(model_dir,
                            f"model_at_{int(target_accuracy * 100)}")
            torch.save(model.state_dict(), cap_path)
            print(f"  Accuracy cap reached: {vaccuracy:.3f} ≥ {target_accuracy:.2f} "
                  f"→ saved '{cap_path}' and stopped training.")
            break

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_path = join(model_dir, "best_model")
            torch.save(model.state_dict(), best_model_path)
            best_model_epoch_file = join(model_dir, "best_model_epoch")
            with open(best_model_epoch_file, "w") as f:
                f.write(str(epoch_number + 1))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

        epoch_number += 1


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
    :param model_dir: path to model directory (contains a file called best_model).
    :param device: device on which to run the model.
    :return: the model.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'resnet18', pretrained=False)
    model.fc = Linear(in_features=512, out_features=2, bias=True)
    #add conditions for when the model does not reached 80% in accuracy
    checkpoint = "model_at_80" if os.path.exists(os.path.join(model_dir, "model_at_80")) \
        else "best_model"

    model.load_state_dict(torch.load(os.path.join(model_dir, checkpoint), weights_only=True))
    # model.load_state_dict(torch.load(os.path.join(model_dir, "best_model"), weights_only=True))
    model.load_state_dict(torch.load(os.path.join(
        model_dir, "model_at_80"), weights_only=True))
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
    dataset_train = get_dataset_transformed(
        db_dir, model_dir, train_dataset_filename)
    dataset_test = get_dataset_transformed(
        db_dir, model_dir, test_dataset_filename)
    dataset_valid = get_dataset_transformed(
        db_dir, model_dir, valid_dataset_filename)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=100, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=100, shuffle=False)

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
