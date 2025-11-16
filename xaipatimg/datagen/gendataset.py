from xaipatimg.datagen.dbimg import load_db
import numpy as np
import csv
import tqdm
import shutil
import os
from sklearn.model_selection import train_test_split

from xaipatimg.datagen.utils import PatImgObj


def create_dataset_based_on_rule(db_dir, datasets_dir_path, csv_filename_train, csv_filename_test, csv_filename_valid,
                                 test_size, valid_size, dataset_pos_samples_nb, dataset_neg_samples_nb, rule_fun,
                                 random_seed=42, filter_on_dim=None, **kwargs):
    """
    Function that creates a training dataset based on the rule that is defined in the rule_fun function. The dataset is
    saved as a csv file and contains a given number of positive and negative samples.
    :param db_dir: path to the root directory of the database.
    :param datasets_dir_path: path to the directory where the datasets will be saved.
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
    :param filter_on_dim: if tuple (xdim, ydim), only the images with this dimension will be considered.
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

        if filter_on_dim is not None:
            if img_content["division"] != filter_on_dim:
                continue

        is_positive, is_excluded = rule_fun(img_content["content"], **kwargs)

        if is_positive:
            if pos_nb < dataset_pos_samples_nb:
                pos_list.append(img_content["path"])
            pos_nb += 1
        elif not is_excluded:
            if neg_nb < dataset_neg_samples_nb:
                neg_list.append(img_content["path"])
            neg_nb += 1

        if pos_nb >= dataset_pos_samples_nb and neg_nb >= dataset_neg_samples_nb:
            break

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
    os.makedirs(datasets_dir_path, exist_ok=True)
    with open(os.path.join(datasets_dir_path, csv_filename_train), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_content_train)
    with open(os.path.join(datasets_dir_path, csv_filename_test), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_content_test)
    with open(os.path.join(datasets_dir_path, csv_filename_valid), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_content_valid)

def extract_sample_from_dataset(db_dir, datasets_dir_path, csv_filename, output_dir_path, pos_samples_nb,
                                neg_samples_nb):
    """
    Function that copies samples from the given dataset in order to visualize the images.
    :param db_dir: path to the root directory of the database.
    :param datasets_dir_path: path to the directory where the datasets will be saved.
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
    with open(os.path.join(datasets_dir_path, csv_filename), "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm.tqdm(csv_reader):
            if row[1] == "1" and pos_nb < pos_samples_nb:
                shutil.copyfile(os.path.join(db_dir, row[0]), os.path.join(pos_dir_path, os.path.basename(row[0])))
                pos_nb += 1
            elif row[1] == "0" and neg_nb < neg_samples_nb:
                shutil.copyfile(os.path.join(db_dir, row[0]), os.path.join(neg_dir_path, os.path.basename(row[0])))
                neg_nb += 1

def _extract_rows_with_only_shape_or_color(img_content, y_division, shape=None, color=None):
    """
    Returns the rows which only contain the given shape or color
    :param img_content: content of the image.
    :param shape: shape to search for.
    :param color: color to search for.
    :param y_division: number of y divisions in the image.
    :return: The rows which only contain the given shape or color, the count of symbols matching the pattern for every
    row, the count of symbols not matching the pattern for every row.
    """
    pattern_counter = np.zeros(y_division,)
    non_pattern_counter = np.zeros(y_division,)

    for c in img_content:
        if shape is not None :
            if c["shape"] == shape:
                pattern_counter[c["pos"][1]] += 1
                continue

        if color is not None:
            if c["color"] == color:
                pattern_counter[c["pos"][1]] += 1
                continue

        non_pattern_counter[c["pos"][1]] += 1

    return np.logical_and(pattern_counter >= 1, non_pattern_counter == 0), pattern_counter, non_pattern_counter

def _extract_cols_with_only_shape_or_color(img_content, x_division, shape=None, color=None):
    """
    Returns the col which only contain the given shape or color
    :param img_content: content of the image.
    :param shape: shape to search for.
    :param color: color to search for.
    :param x_division: number of x divisions in the image.
    :return:
    """
    pattern_counter = np.zeros(x_division,)
    non_pattern_counter = np.zeros(x_division,)

    for c in img_content:
        if shape is not None :
            if c["shape"] == shape:
                pattern_counter[c["pos"][0]] += 1
                continue

        if color is not None:
            if c["color"] == color:
                pattern_counter[c["pos"][0]] += 1
                continue

        non_pattern_counter[c["pos"][0]] += 1

    return np.logical_and(pattern_counter >= 1, non_pattern_counter == 0)

def generic_rule_exist_row_with_only_shape(img_content, shape, y_division):
    """
    Returns True iff there is at least one row in the given image that only contains the given shape.
    :param img_content: dictionary content of the image.
    :param shape: shape to identify.
    :param y_division: number of y divisions.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    match, _, _ = _extract_rows_with_only_shape_or_color(img_content, y_division, shape=shape)
    return np.sum(match) >= 1, False

def generic_rule_exist_row_with_only_color(img_content, color, y_division):
    """
    Returns True iff there is at least one row in the given image that only contains the given color.
    :param img_content: dictionary content of the image.
    :param color: color to identify.
    :param y_division: number of y divisions.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    match, _, _ = _extract_rows_with_only_shape_or_color(img_content, y_division, color=color)
    return np.sum(match) >= 1, False

def generic_rule_exist_column_with_only_shape(img_content, shape, x_division):
    """
    Returns True iff there is at least one row in the given image that only contains the given shape.
    :param img_content: dictionary content of the image.
    :param shape: shape to identify.
    :param x_division: number of y divisions.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    return np.sum(_extract_cols_with_only_shape_or_color(img_content, x_division, shape=shape)) >= 1, False

def generic_rule_exist_row_with_only_color_and_col_with_only_shape(img_content, color, shape, x_division, y_division):
    """
    Returns True iff there is at least one row in the given image that only contains the given color and at least
    one column in the given image that only contains the given shape.
    :param img_content: dictionary content of the image.
    :param color: color to identify.
    :param shape: shape to identify.
    :param x_division: number of x divisions.
    :param y_division: number of y divisions.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    return generic_rule_exist_row_with_only_color(img_content, color, y_division) \
        and generic_rule_exist_column_with_only_shape(img_content, shape, x_division), False

def generic_rule_N_times_color_exactly(img_content, N, color, x_division, y_division):
    """
    Returns True iff there is exactly N times the given color in the image.
    :param img_content: dictionary content of the image.
    :param N: number of instances of the given color to search for.
    :param color: color to count the instances of.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    obj = PatImgObj({"content": img_content, "division": (x_division, y_division), "path": None, "size": None})
    return len(obj.get_symbols_by(color=color)) == N, False

def generic_rule_N_times_color_shape_exactly(img_content, N, color, shape, x_division, y_division,
                                             restrict_plus_minus_1=False):
    """
    Returns True iff there is exactly N times the symbol with given color and shape in the image.
    :param img_content: dictionary content of the image.
    :param N: number of instances of the given color to search for.
    :param color: color of the symbol to count the instances of.
    :param shape: shape of the symbol to count the instances of.
    :param x_division: number of x divisions.
    :param y_division: number of y divisions.
    :param restrict_plus_minus_1: If true, an exclusion criteria is applied which only includes images which contain
    [N-1, N+1] symbols of the given color and shape.
    :return: respects rule, is_excluded
    """
    obj = PatImgObj({"content": img_content, "division": (x_division, y_division), "path": None, "size": None})

    count = len(obj.get_symbols_by(color=color, shape=shape))
    exclusion = (count < N - 1 or count > N + 1) if restrict_plus_minus_1 else False
    return  count == N, exclusion

def generic_rule_shape_color_plus_shape_equals_N(img_content, shape1, color1, shape2, N, x_division, y_division):
    """
    Return true iff the number of instances of the given color and shape plus the number of instances of the given
    shape equals the given N value.
    :param img_content: dictionary content of the image.
    :param shape1: shape of the first element to count.
    :param color1: color of the first element to count.
    :param shape2: shape of the second element to count.
    :param N: integer value to compare the sum with.
    :param x_division: number of x divisions.
    :param y_division: number of y divisions.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    obj = PatImgObj({"content": img_content, "division": (x_division, y_division), "path": None, "size": None})
    return len(obj.get_symbols_by(shape=shape1, color=color1)) + len(obj.get_symbols_by(shape=shape2)) == N, False

def generic_rule_shape_color_times_2_shape_equals_shape(img_content, shape1, color1, shape2, x_division, y_division):
    """
    Return true iff the number of instances of the given color and shape multiplied by 2 equals to the number of
    instances of the given shape
    the given N value.
    :param img_content: dictionary content of the image.
    :param shape1: shape of the first element to count.
    :param color1: color of the first element to count.
    :param shape2: shape of the second element to count.
    :param x_division: number of x divisions.
    :param y_division: number of y divisions.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    obj = PatImgObj({"content": img_content, "division": (x_division, y_division), "path": None, "size": None})
    return len(obj.get_symbols_by(shape=shape1, color=color1)) * 2 == len(obj.get_symbols_by(shape=shape2)), False

def generic_rule_shape_color_even(img_content, color, shape, x_division, y_division):
    """
    Return true iff the number of symbols of the given color and shape is evnen
    :param img_content: dictionary content of the image.
    :param color: color
    :param shape: shape
    :param x_division: number of x divisions.
    :param y_division: number of y divisions.
    :return: is_excluded (no exclusion criteria)
    """
    obj = PatImgObj({"content": img_content, "division": (x_division, y_division), "path": None, "size": None})
    return len(obj.get_symbols_by(shape=shape, color=color)) % 2 == 0, False

def generic_rule_shape_in_every_row(img_content, shape, y_division):
    """
    Return True iff there is the given shape in every row of the image.
    :param img_content: dictionary content of the image.
    :param shape: shape to identify.
    :param y_division: number of y divisions.
    :return: respects rule, is_excluded (no exclusion criteria)
    """
    _, pattern_count, _ = _extract_rows_with_only_shape_or_color(img_content, y_division, shape=shape)
    return np.all(pattern_count), False


def generic_rule_pattern_exactly_1_time_exclude_more(img_content, pattern_content, x_division_full, y_division_full,
                                                     x_division_pattern, y_division_pattern, consider_rotations=False):
    """
    Returns True iff the given pattern (defined as a subimage) appears exactly 1 time in the given full image.
    Excludes the images where the pattern appears more than once. If consider_rotations is True, the function also
    searches for left and right rotations of the pattern.
    :param img_content: dictionary content of the image.
    :param pattern_content: dictionary content of the pattern to search for
    :param x_division_full: X division of the full image
    :param y_division_full: Y division of the full image
    :param x_division_pattern: X division of the pattern to search for
    :param y_division_pattern: Y division of the pattern to search for
    :param consider_rotations: whether to consider rotations of the pattern when searching (left and right rotations only).
    :return: respects rule, is_excluded
    """

    obj = PatImgObj({"content": img_content, "division": (x_division_full, y_division_full), "path": None, "size": None})
    submatrixes_positions = obj.find_submatrix_positions(pattern_content, (x_division_pattern, y_division_pattern),
                                                         consider_rotations=consider_rotations)

    return len(submatrixes_positions) == 1, len(submatrixes_positions) > 1


def generic_rule_pattern_exactly_N_times(img_content, pattern_content, N, x_division_full, y_division_full,
                                         x_division_pattern, y_division_pattern):
    """
    Returns True iff the given pattern (defined as a subimage) appears exactly N time in the given full image.
    No exclusion criteria.
    :param img_content: dictionary content of the image.
    :param pattern_content: dictionary content of the pattern to search for
    :param N: integer value to compare the sum with.
    :param x_division_full: X division of the full image
    :param y_division_full: Y division of the full image
    :param x_division_pattern: X division of the pattern to search for
    :param y_division_pattern: Y division of the pattern to search for
    """

    obj = PatImgObj({"content": img_content, "division": (x_division_full, y_division_full), "path": None, "size": None})
    submatrixes_positions = obj.find_submatrix_positions(pattern_content, (x_division_pattern, y_division_pattern))

    return len(submatrixes_positions) == N, False


def create_dataset_generic_rule_extract_sample(db_dir, datasets_dir_path, csv_name_train, csv_name_test, csv_name_valid,
                                               test_size, valid_size, dataset_pos_samples_nb, dataset_neg_samples_nb,
                                               sample_path, sample_nb_per_class, generic_rule_fun, filter_on_dim=None,
                                               **kwargs):
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
    :param filter_on_dim: if tuple (xdim, ydim), only samples with given dimension will be considered.
    :param kwargs: kwargs to give to the generic rule function.
    :return:
    """

    # Dataset generation
    create_dataset_based_on_rule(db_dir, datasets_dir_path, csv_name_train, csv_name_test, csv_name_valid,
                                 test_size=test_size, valid_size=valid_size,
                                 dataset_pos_samples_nb=dataset_pos_samples_nb,
                                 dataset_neg_samples_nb=dataset_neg_samples_nb,
                                 rule_fun=generic_rule_fun, filter_on_dim=filter_on_dim, **kwargs)

    # Sample extraction
    extract_sample_from_dataset(db_dir, datasets_dir_path, csv_name_train, sample_path, sample_nb_per_class,
                                sample_nb_per_class)
