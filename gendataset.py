from dbimg import load_db
import numpy as np
import csv
import tqdm
import shutil
import os

def create_dataset_based_on_rule(db_path, dataset_path, dataset_pos_samples_nb, dataset_neg_samples_nb, rule_fun,
                                 random_seed=42):
    """
    Function that creates a training dataset based on the rule that is defined in the rule_fun function. The dataset is
    saved as a csv file and contains a given number of positive and negative samples.
    :param db_path: path to the database in json format.
    :param dataset_path: path to the output dataset in csv format.
    :param dataset_pos_samples_nb: number of positive samples to be contained in the dataset.
    :param dataset_neg_samples_nb: number of negative samples to be contained in the dataset.
    :param rule_fun: boolean function that defines whether the given image content is positive or negative.
    :param random_seed: seed which is used for dataset random ordering.
    :return: None
    """
    # Load and shuffle images content
    db = load_db(db_path)
    img_content_list = list(db.values())
    np.random.seed(random_seed)
    np.random.shuffle(img_content_list)

    pos_list = []
    neg_list = []
    pos_nb = 0
    neg_nb = 0

    # Extracting positive and negative samples
    for img_content in tqdm.tqdm(img_content_list):
        is_positive = rule_fun(img_content["content"])
        if is_positive and pos_nb < dataset_pos_samples_nb:
            pos_list.append(img_content["path"])
            pos_nb += 1
        elif not is_positive and neg_nb < dataset_neg_samples_nb:
            neg_list.append(img_content["path"])
            neg_nb += 1

    if pos_nb != dataset_pos_samples_nb or neg_nb != dataset_neg_samples_nb:
        raise RuntimeError("Could not extract enough positive (" + str(pos_nb) + "/" + str(dataset_pos_samples_nb) +
                           ") or negative (" + str(neg_nb) + "/" + str(dataset_neg_samples_nb) +") samples.")

    # Forming CSV content
    pos_class_vect = np.full(pos_nb, 1)
    neg_class_vect = np.full(neg_nb, 0)
    csv_content = np.array([np.concatenate((pos_list, neg_list), axis=0),
                            np.concatenate((pos_class_vect, neg_class_vect), axis=0)]).T

    # Writing dataset to CSV file
    with open(dataset_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_content)

def extract_sample_from_dataset(dataset_path, output_dir_path, pos_samples_nb, neg_samples_nb):
    """
    Function that copies samples from the given dataset in order to visualize the images.
    :param dataset_path: path to the dataset in csv format.
    :param output_dir_path: path where to copy the image samples.
    :param pos_samples_nb: number of positive samples to extract.
    :param neg_samples_nb: number of negative samples to extract.
    :return: None
    """
    pos_nb = 0
    neg_nb = 0
    pos_dir_path = os.path.join(output_dir_path, "positive")
    neg_dir_path = os.path.join(output_dir_path, "negative")

    # Create directories if they don't exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    if not os.path.exists(pos_dir_path):
        os.makedirs(pos_dir_path)
    if not os.path.exists(neg_dir_path):
        os.makedirs(neg_dir_path)

    # Copy files
    with open(dataset_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm.tqdm(csv_reader):
            if row[1] == "1" and pos_nb < pos_samples_nb:
                shutil.copyfile(row[0], os.path.join(pos_dir_path, os.path.basename(row[0])))
                pos_nb += 1
            elif row[1] == "0" and neg_nb < neg_samples_nb:
                shutil.copyfile(row[0], os.path.join(neg_dir_path, os.path.basename(row[0])))
                neg_nb += 1