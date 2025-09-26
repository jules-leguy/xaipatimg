from xaipatimg.datagen.dbimg import load_db
import numpy as np
import csv
import tqdm
import shutil
import os
from sklearn.model_selection import train_test_split

def create_dataset_based_on_rule(db_dir, csv_filename_train, csv_filename_test, csv_filename_valid, test_size,
                                 valid_size, dataset_pos_samples_nb, dataset_neg_samples_nb, rule_fun, random_seed=42, **kwargs):
    """
    Function that creates a training dataset based on the rule that is defined in the rule_fun function. The dataset is
    saved as a csv file and contains a given number of positive and negative samples.
    :param db_dir: path to the root directory of the database.
    :param csv_filename_train: name of the csv file that contains the training dataset.
    :param csv_filename_test: name of the csv file that contains the testing dataset.
    :param csv_filename_valid: name of the csv file that contains the validation dataset.
    :param test_size: number of samples or proportion of the data to be put in the validation set.
    :param valid_size: number of samples or proportion of the dataset to be put in the validation set (in proportion of
    the data that has not been put in the test set).
    :param dataset_pos_samples_nb: number of positive samples to be contained in the dataset.
    :param dataset_neg_samples_nb: number of negative samples to be contained in the dataset.
    :param rule_fun: boolean function that defines whether the given image content is positive or negative.
    :param random_seed: seed which is used for dataset random ordering.
    :return: None
    """
    # Load and shuffle images content
    db = load_db(db_dir)
    img_content_list = list(db.values())
    np.random.seed(random_seed)
    np.random.shuffle(img_content_list)

    pos_list = []
    neg_list = []
    pos_nb = 0
    neg_nb = 0

    # Extracting positive and negative samples
    for img_content in tqdm.tqdm(img_content_list):
        is_positive = rule_fun(img_content["content"], **kwargs)
        if is_positive:
            if pos_nb < dataset_pos_samples_nb:
                pos_list.append(img_content["path"])
            pos_nb += 1
        else:
            if neg_nb < dataset_neg_samples_nb:
                neg_list.append(img_content["path"])
            neg_nb += 1

    print("Total number of positive instances found in database : " + str(pos_nb))
    print("Total number of negative instances found in database : " + str(neg_nb))

    if len(pos_list) != dataset_pos_samples_nb or len(neg_list) != dataset_neg_samples_nb:
        raise RuntimeError("Could not extract enough positive (" + str(pos_nb) + "/" + str(dataset_pos_samples_nb) +
                           ") or negative (" + str(neg_nb) + "/" + str(dataset_neg_samples_nb) +") samples.")

    # Forming dataset content
    y = np.concatenate((np.full(len(pos_list), 1), np.full(len(neg_list), 0)), axis=0)
    img_list = np.concatenate((pos_list, neg_list), axis=0)

    # Making sure all the data is unique before splitting it into train/test/valid sets
    assert len(np.unique(img_list)) == len(img_list)

    # Splitting training, testing and validation sets
    img_list_nontest, img_list_test, y_nontest, y_test = train_test_split(img_list, y,
                                                                          test_size=test_size, random_state=random_seed,
                                                                          stratify=y)
    img_list_train, img_list_valid, y_train, y_valid = train_test_split(img_list_nontest, y_nontest,
                                                                        test_size=valid_size, random_state=random_seed,
                                                                        stratify=y_nontest)

    csv_content_train = np.array([np.concatenate((["path"], img_list_train), axis=0),
                                  np.concatenate((["class"], y_train), axis=0)]).T

    csv_content_test = np.array([np.concatenate((["path"], img_list_test), axis=0),
                                 np.concatenate((["class"], y_test), axis=0)]).T

    csv_content_valid = np.array([np.concatenate((["path"], img_list_valid), axis=0),
                                  np.concatenate((["class"], y_valid), axis=0)]).T

    # Writing dataset to CSV files
    os.makedirs(os.path.join(db_dir, "datasets"), exist_ok=True)
    with open(os.path.join(db_dir, "datasets", csv_filename_train), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_content_train)
    with open(os.path.join(db_dir, "datasets", csv_filename_test), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_content_test)
    with open(os.path.join(db_dir, "datasets", csv_filename_valid), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_content_valid)

def extract_sample_from_dataset(db_dir, csv_filename, output_dir_path, pos_samples_nb, neg_samples_nb):
    """
    Function that copies samples from the given dataset in order to visualize the images.
    :param db_dir: path to the root directory of the database.
    :param csv_filename: name of the csv file contained in the database folder.
    :param output_dir_path: path where to save the sample of the dataset.
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
    with open(os.path.join(db_dir, "datasets", csv_filename), "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm.tqdm(csv_reader):
            if row[1] == "1" and pos_nb < pos_samples_nb:
                shutil.copyfile(os.path.join(db_dir, row[0]), os.path.join(pos_dir_path, os.path.basename(row[0])))
                pos_nb += 1
            elif row[1] == "0" and neg_nb < neg_samples_nb:
                shutil.copyfile(os.path.join(db_dir, row[0]), os.path.join(neg_dir_path, os.path.basename(row[0])))
                neg_nb += 1

def _extract_rows_with_only_shape(img_content, shape, y_division):
    """
    Returns the row which only contain the given shape
    :param img_content: content of the image.
    :param shape: shape to search for.
    :param y_division: number of y divisions in the image.
    :return:
    """
    circles_counter = np.zeros(y_division,)
    non_circles_counter = np.zeros(y_division,)

    for c in img_content:
        if c["shape"] == shape:
            circles_counter[c["pos"][1]] += 1
        else:
            non_circles_counter[c["pos"][1]] += 1

    return np.logical_and(circles_counter >= 1, non_circles_counter == 0)

def generic_rule_exist_row_with_only_shape(img_content, shape, y_division):
    """
    Returns True iff there is at least one row in the given image that only contains the given shape.
    :param img_content: dictionary content of the image.
    :param shape: shape to identify.
    :param y_division: number of y divisions.
    :return:
    """
    return np.sum(_extract_rows_with_only_shape(img_content, shape, y_division)) >= 1


def generic_rule_N_times_color_exactly(img_content, N, color):
    """
    Returns True iff there is exactly N times the given color in the image.
    :param img_content: dictionary content of the image.
    :param N: number of instances of the given color to search for.
    :param color: color to count the instances of.
    :return:
    """
    color_counter = 0
    for c in img_content:
        if c["color"] == color:
            color_counter += 1

    return color_counter == N



def create_dataset_generic_rule_extract_sample(db_dir, csv_name_train, csv_name_test, csv_name_valid,
                                               test_size, valid_size, dataset_pos_samples_nb, dataset_neg_samples_nb,
                                               sample_path, sample_nb_per_class, generic_rule_fun, **kwargs):
    """
    Creating a dataset with the given characteristics and following the given generic rule, and extracting a sample
    of positive and negative instances.

    :param db_dir: path to the root directory of the database.
    :param csv_name_train: path to the csv file indexing training dataset.
    :param csv_name_test: path to the csv file indexing test dataset.
    :param csv_name_valid: path to the csv file indexing validation dataset.
    :param test_size: size of the test dataset.
    :param valid_size: size of the validation dataset.
    :param dataset_pos_samples_nb: number of positive samples in the full dataset.
    :param dataset_neg_samples_nb: number of negative samples in the full dataset.
    :param sample_path: directory where to save the sample of the dataset.
    :param sample_nb_per_class: size of the sample for each class.
    :param generic_rule_fun: generic rule function that is used to generate the dataset.
    :param kwargs: kwargs to give to the generic rule function.
    :return:
    """

    # Dataset generation
    create_dataset_based_on_rule(db_dir, csv_name_train, csv_name_test, csv_name_valid,
                                 test_size=test_size, valid_size=valid_size,
                                 dataset_pos_samples_nb=dataset_pos_samples_nb,
                                 dataset_neg_samples_nb=dataset_neg_samples_nb,
                                 rule_fun=generic_rule_fun, **kwargs)

    # Sample extraction
    extract_sample_from_dataset(db_dir, csv_name_train, sample_path, sample_nb_per_class, sample_nb_per_class)
    # %%
