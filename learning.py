import pandas as pd
import os
import torch
from PIL import Image

class XAIGenImgDataset(torch.utils.data.Dataset):
    def __init__(self, database_root_path, csv_dataset_filename, transform=None, target_transform=None):
        """
        Initialize the dataset object.
        :param database_root_path: path to the root directory of the database
        :param csv_dataset_filename: name of the csv dataset file
        :param transform: pipeline for transforming input data
        :param target_transform: pipeline for transforming labels
        """
        self.database_root_path = database_root_path
        dataset_csv = pd.read_csv(os.path.join(database_root_path, csv_dataset_filename))
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
        img_path = os.path.join(self.database_root_path, self.img_list[idx])
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
        return os.path.join(self.database_root_path, self.img_list[idx])

