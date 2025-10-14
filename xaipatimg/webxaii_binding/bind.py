import csv
import json
import os
import random
import shutil

import numpy as np
from sklearn.model_selection import train_test_split

def _create_task_desc_json(res_dir, name, question):
    """
    Creating the JSON file that describes the task (question and examples).
    :param res_dir: resources directory.
    :param name: name of the task.
    :param question: text question.
    :return:
    """
    json_data = {
        "question": question,
        "pos_example": f"assets/res/input/{name}/pos_example.png",
        "neg_example": f"assets/res/input/{name}/neg_example.png",
    }
    with open(os.path.join(res_dir, "tasks", name + ".json"), 'w') as f:
        json.dump(json_data, f)

def _create_content_csv(res_dir, idx_selected, y, y_pred, name, XAI_col_path_dict):
    """
    Creating the content.csv file that contains the y, y_pred values as well as paths to images for the given task.
    :param res_dir: resources directory.
    :param idx_selected: vector of indices of the samples selected.
    :param y: vector of labels of the samples selected.
    :param y_pred: vector of predicted labels of the samples selected.
    :param name: name of the task.
    :param XAI_col_path_dict: dictionary that maps the names of XAI techniques to the vector of paths.
    :return:
    """
    csv_data = []

    img_dir = os.path.join("res", "input", name)
    AI_dir = os.path.join("res", "AI", name)

    for row_idx in range(len(idx_selected)):
        curr_row_dict = {"target": y[row_idx], "pred": y_pred[row_idx], "og_idx": idx_selected[row_idx],
                         "source": os.path.join(img_dir, f"{idx_selected[row_idx]}.png"),
                         "AI": os.path.join(AI_dir, f"{idx_selected[row_idx]}.png")}

        for XAI_key, XAI_paths in XAI_col_path_dict.items():
            curr_row_dict["xai_"+XAI_key] = f"{os.path.join("res", "xai_"+XAI_key, name, os.path.basename(XAI_paths[row_idx]))}"

        csv_data.append(curr_row_dict)

    with open(os.path.join(res_dir, "tasks", name + "_content.csv"), 'w', newline='') as csvfile:
        fieldnames = ["target", "pred", "og_idx", "source", "AI"] + ["xai_" + v for v in XAI_col_path_dict.keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)


def _copy_images(db, model_dir, res_dir, name, idx_selected, img_paths, AI_paths, XAI_col_path_dict, pos_example_path,
                 neg_example_path):
    """
    Copy all images for the given task (source images, AI only images, XAI images) from the database and model folders
    to the resources directory.
    :param db: database directory.
    :param model_dir: model directory.
    :param res_dir: resources directory.
    :param name: name of the task.
    :param idx_selected: vector of indices of the samples selected.
    :param img_paths: list of paths to source images.
    :param AI_paths: list of paths to AI only images.
    :param XAI_col_path_dict: dictionary that maps the names of XAI techniques to the vector of paths.
    :return:
    """
    img_dir = os.path.join(res_dir, "input", name)
    AI_dir = os.path.join(res_dir, "AI", name)
    os.makedirs(img_dir, exist_ok = True)
    os.makedirs(AI_dir, exist_ok = True)
    for XAI_key, XAI_paths in XAI_col_path_dict.items():
        os.makedirs(os.path.join(res_dir, "xai_" + XAI_key), exist_ok=True)
        os.makedirs(os.path.join(res_dir, "xai_" + XAI_key, name), exist_ok=True)

    # Copying images for all selected samples
    for idx in range(len(idx_selected)):
        shutil.copyfile(os.path.join(db, img_paths[idx]),
                        os.path.join(img_dir, str(idx_selected[idx]) + ".png"))
        shutil.copyfile(os.path.join(model_dir, name, AI_paths[idx]),
                        os.path.join(AI_dir, str(idx_selected[idx]) + ".png"))

        for XAI_key, XAI_paths in XAI_col_path_dict.items():
            shutil.copyfile(os.path.join(model_dir, name, XAI_paths[idx]),
                            os.path.join(res_dir, "xai_" + XAI_key, name, str(idx_selected[idx]) + ".png"))

    # Copying images for examples
    shutil.copyfile(os.path.join(db, pos_example_path), os.path.join(img_dir, "pos_example.png"))
    shutil.copyfile(os.path.join(db, neg_example_path), os.path.join(img_dir, "neg_example.png"))


def _create_res_task(db_dir, model_dir, res_dir, name, question, pos_example_path, neg_example_path, idx_selected, y,
                     y_pred, img_paths, AI_paths, XAI_col_path_dict):
    """
    Create and import all the resources for the given task.
    :param db_dir: database directory.
    :param model_dir: model directory.
    :param res_dir: resources directory.
    :param name: name of the task.
    :param question: question text for the given task.
    :param pos_example_path : path to a positive example of image for the given task.
    :param neg_example_path : path to a negative example of image for the given task.
    :param idx_selected: indices of the samples selected.
    :param y: vector of labels of the samples selected.
    :param y_pred: vector of predicted labels of the samples selected.
    :param img_paths: list of paths to source images.
    :param AI_paths: list of paths to AI only images.
    :param XAI_col_path_dict: dictionary that maps the names of XAI techniques to the vector of paths.
    :return:
    """
    _create_content_csv(res_dir, idx_selected, y, y_pred, name, XAI_col_path_dict)
    _create_task_desc_json(res_dir, name, question)
    _copy_images(db_dir, model_dir, res_dir, name, idx_selected, img_paths, AI_paths, XAI_col_path_dict,
                 pos_example_path, neg_example_path)

def _select_examples(data_d, idx_to_consider):
    """
    Selecting examples from the dataset.
    :param data_d: data dictionary.
    :param idx_to_consider: indices that are considered to search for examples.
    :return: pos_example_path (path to a positive example), neg_example_path (path to a negative example).
    """

    pos_example_path = None
    neg_example_path = None

    for idx in idx_to_consider:
        if data_d["y"][idx] == 1:
            pos_example_path = data_d["path"][idx]
            break

    for idx in idx_to_consider:
        if data_d["y"][idx] == 0:
            neg_example_path = data_d["path"][idx]
            break

    return pos_example_path, neg_example_path

def _is_error(y, y_pred):
    return y_pred != y

def _is_pos(y):
    return y == 1

def _sample_instances(y, y_pred, sample_size, shown_accuracy):
    """
    Performing the sampling of data.
    :param y: vector of true labels.
    :param y_pred: vector of predicted labels.
    :param sample_size: size of the sample to generate.
    :param shown_accuracy: proportion of positive samples in the generated set.
    :return: index of samples selected, index of samples not selected.
    """

    # Calculating the number of positive and negative samples (If uneven number, the majority class is selected
    # randomly).
    if random.random() < 0.5:
        n_pos_target = sample_size // 2
        n_neg_target = sample_size - n_pos_target
    else:
        n_neg_target = sample_size // 2
        n_pos_target = sample_size - n_neg_target

    # Calculating the number of correct samples and errors
    n_correct_target = round(sample_size*shown_accuracy)
    n_errors_target = sample_size-n_correct_target

    n_pos_selected = 0
    n_neg_selected = 0
    n_errors_selected = 0

    # Generating random index of all samples
    all_samples_idx = np.arange(len(y))
    np.random.shuffle(all_samples_idx)

    idx_selected = []

    # Iterating over the samples to extract the model errors instances
    for idx in all_samples_idx:
        if _is_error(y[idx], y_pred[idx]):
            idx_selected.append(idx)

            n_errors_selected += 1
            if _is_pos(y[idx]):
                n_pos_selected += 1
            else:
                n_neg_selected += 1

        if n_errors_selected == n_errors_target:
            break

    if n_errors_selected != n_errors_target:
        raise ValueError(f"Impossible to extract the expected number of model errors ({n_errors_selected}/"
                         f"{n_errors_target}) to reach the target shown accuracy ({shown_accuracy}).")

    # Iterating again to extract the rest of instances (which are correctly predicted)
    for idx in all_samples_idx:
        if not _is_error(y[idx], y_pred[idx]):
            if _is_pos(y[idx]):
                if n_pos_selected < n_pos_target:
                    idx_selected.append(idx)
                    n_pos_selected += 1
            else:
                if n_neg_selected < n_neg_target:
                    idx_selected.append(idx)
                    n_neg_selected += 1
        if n_pos_selected == n_pos_target and n_neg_selected == n_neg_target:
            break

    if n_pos_selected != n_pos_target or n_neg_selected != n_neg_target:
        raise ValueError(f"Impossible to extract the number of positive samples ({n_pos_selected}/{n_pos_target}) or"
                         f"negative samples ({n_neg_selected}/{n_neg_target}).)")

    # Shuffling the order of selected instances (to not have the model errors first)
    idx_selected = np.array(idx_selected)
    np.random.shuffle(idx_selected)

    idx_not_selected = np.setdiff1d(all_samples_idx, idx_selected)

    return idx_selected, idx_not_selected

def generate_resources_dir(db_dir, interface_dir, model_dir, models_names_list, tasks_q_list, XAI_names_list,
                           shown_accuracy_list, samples_nb_interface_list, random_seed=42):
    """
    Generating the resources directory which is used in the experimental interface (WebXAII).

    :param db_dir: path to the database directory.
    :param interface_dir: path to the interface directory where the resources folder will be saved.
    :param model_dir: path to the directory which contains the models.
    :param models_names_list: list of model names.
    :param tasks_q_list: list of task questions.
    :param XAI_names_list: list of XAI techniques as saved in the models folders.
    :param shown_accuracy_list: list of target accuracy that was used when training the models.
    :param samples_nb_interface_list: list of number of samples to generate for the interface for each model.
    :param random_seed: random seed used for sampling.
    :return:
    """

    # Random seeding
    np.random.seed(random_seed)
    random.seed(random_seed)

    os.makedirs(interface_dir, exist_ok=True)
    res_dir = os.path.join(interface_dir, "res")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(os.path.join(res_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "tasks"), exist_ok=True)

    # Iterating over all the model names
    for model_idx, model_name in enumerate(models_names_list):
        data_d = {}
        with open(os.path.join(model_dir, model_name, "xai_index.csv"), 'r') as f:
            fieldnames = ["idx", "y", "y_pred", "path", "AI"] + XAI_names_list
            reader = csv.DictReader(f, delimiter=',', fieldnames=fieldnames)

            for idx_row, row in enumerate(reader):
                for fieldname in fieldnames:
                    if idx_row == 0:
                        data_d[fieldname] = []
                    else:
                        if fieldname in ["idx", "y", "y_pred"]:
                            data_d[fieldname].append(int(row[fieldname]))
                        else:
                            data_d[fieldname].append(row[fieldname])

            for fieldname in fieldnames:
                data_d[fieldname] = np.array(data_d[fieldname])

            sample_size = samples_nb_interface_list[model_idx]
            shown_accuracy = shown_accuracy_list[model_idx]
            idx_selected, idx_not_selected = _sample_instances(data_d["y"], data_d["y_pred"], sample_size,
                                                               shown_accuracy)

            # Extracting a positive example and a negative example (which are not part of the sampled data)
            pos_example_path, neg_example_path = _select_examples(data_d, idx_not_selected)

            XAI_col_path_dict = {name: data_d[name][idx_selected] for name in XAI_names_list}

            _create_res_task(db_dir, model_dir, res_dir, model_name, tasks_q_list[model_idx], pos_example_path,
                             neg_example_path, idx_selected,data_d["y"][idx_selected], data_d["y_pred"][idx_selected],
                             data_d["path"][idx_selected], data_d["AI"][idx_selected], XAI_col_path_dict)
