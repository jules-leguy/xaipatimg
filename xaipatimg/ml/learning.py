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


class PatImgDataset(torch.utils.data.Dataset):
    def __init__(self, db_dir, csv_dataset_filename, transform=None, target_transform=None):
        """
        Initialize the dataset object.
        :param db_dir: path to the root directory of the database
        :param csv_dataset_filename: name of the csv dataset file
        :param transform: pipeline for transforming input data
        :param target_transform: pipeline for transforming labels
        """
        self.db_dir = db_dir
        dataset_csv = pd.read_csv(os.path.join(db_dir, "datasets", csv_dataset_filename))
        self.img_list = dataset_csv["path"]
        self.img_labels = dataset_csv["class"]
        self.transform = transform
        self.target_transform = target_transform

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
        img_path = os.path.join(self.db_dir, self.img_list[idx])
        image = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
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
    dataset_no_norm = PatImgDataset(db_dir, dataset_filename, transform=preprocess_no_norm)
    alldata_no_norm = np.array([dataset_no_norm[i][0] for i in tqdm.tqdm(range(len(dataset_no_norm)))])
    means = [np.mean(x).astype(float) for x in [alldata_no_norm[:, channel, :] for channel in range(alldata_no_norm.shape[1])]]
    stds = [np.std(x).astype(float) for x in [alldata_no_norm[:, channel, :] for channel in range(alldata_no_norm.shape[1])]]
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
                         lr_gamma=0.1, train_loss_write_period_logs=100):
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
    preprocess_no_norm = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    means, stds = compute_mean_std_dataset(db_dir, train_dataset_filename, preprocess_no_norm)

    # Writing training set statistics to file
    with open(os.path.join(model_dir, "train_set_stats.json"), "w") as f:
        json.dump({"mean": means, "std": stds}, f)

    print("Train dataset statistics : " + str(means) + " " + str(stds))

    # Definition of complete preprocessing pipeline that includes normalization based on the training data
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    # Importing the data
    dataset_train = PatImgDataset(db_dir, train_dataset_filename, transform=preprocess)
    dataset_valid = PatImgDataset(db_dir, valid_dataset_filename, transform=preprocess)
    training_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    # Creating the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
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

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_path = join(model_dir, "best_model")
            torch.save(model.state_dict(), best_model_path)
            best_model_epoch_file = join(model_dir, "best_model_epoch")
            with open(best_model_epoch_file, "w") as f:
                f.write(str(epoch_number + 1))

        epoch_number += 1

def _compute_scores(data_loader, model, device):
    probabilities = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs.float())
            probabilities.append(torch.nn.functional.softmax(outputs, dim=1).tolist())
            labels.append(labels.tolist())

    y_pred = np.round(probabilities).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_pred, labels).ravel()

    return {
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision_score(labels, y_pred),
        "recall": recall_score(labels, y_pred),
        "roc_auc": roc_auc_score(labels, probabilities),
        "confusion matrix": {
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        }
    }

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

    with open(os.path.join(model_dir, "train_set_stats.json"), "r") as f:
        stats_dict = json.load(f)

    means, stds = stats_dict["means"], stats_dict["stds"]

    # Definition of complete preprocessing pipeline that includes normalization based on the training data
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    # Importing the data
    dataset_train = PatImgDataset(db_dir, train_dataset_filename, transform=preprocess)
    dataset_test = PatImgDataset(db_dir, test_dataset_filename, transform=preprocess)
    dataset_valid = PatImgDataset(db_dir, valid_dataset_filename, transform=preprocess)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=100, shuffle=False)

    # Creating the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = Linear(in_features=512, out_features=2, bias=True)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model"), weights_only=True))
    model.eval()
    model = model.to(device)

    # Computing the scores
    results = {
        "train" : _compute_scores(train_loader, model, device),
        "test" : _compute_scores(test_loader, model, device),
        "valid" : _compute_scores(valid_loader, model, device)
    }

    # Writing the results
    with open(join(model_dir, "results.json"), "w") as f:
        json.dump(results, f)

    print(results)
    return results