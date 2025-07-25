import os
import pathlib

import cv2
import shap
from pytorch_grad_cam import GradCAM
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from xaipatimg.ml import resnet18_preprocess_no_norm
from xaipatimg.ml.learning import load_resnet18_based_model, get_dataset_transformed, make_prediction
import numpy as np
import io
from xaipatimg.ml.utils import nhwc_to_nchw, nchw_to_nhwc, add_margin, crop_white_border, add_border
import tqdm
from joblib import Parallel, delayed


def _create_dirs(xai_output_path):
    """
    Creating directories for the output of XAI methods. Creating the main directory and the directories for true
    positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
    :param xai_output_path: path of the directory where XAI output will be saved.
    :return:
    """
    os.makedirs(xai_output_path, exist_ok=True)
    os.makedirs(os.path.join(xai_output_path, "TP"), exist_ok=True)
    os.makedirs(os.path.join(xai_output_path, "TN"), exist_ok=True)
    os.makedirs(os.path.join(xai_output_path, "FP"), exist_ok=True)
    os.makedirs(os.path.join(xai_output_path, "FN"), exist_ok=True)


def _get_subfolder(pred, true):
    """
    Getting the subfolder name depending on the prediction.
    :param pred: predicted class.
    :param true: true class.
    :return: str that represents the subfolder name.
    """
    if true == 1 and pred == 1:
        return "TP"
    elif true == 0 and pred == 0:
        return "TN"
    elif true == 1 and pred == 0:
        return "FN"
    elif true == 0 and pred == 1:
        return "FP"
    return None


def generate_cam_resnet18(cam_technique, db_dir, dataset_filename, model_dir, device="cuda:0"):
    """
    Generating cam images for given model on test set, which are saved into the model directory.
    :param cam_technique: description of the cam technique (str) to use to generate explanations
    :param db_dir: root of the database.
    :param model_dir: path of the model directory.
    :param dataset_filename: filename of the dataset.
    :param device: device to use for pytorch computation.

    :return:
    """

    # Selecting cam method
    if cam_technique.lower() == "gradcam":
        cam_fun = GradCAM
    else:
        raise RuntimeError("Unknown cam technique")

    # Creating directories
    xai_output_path = os.path.join(model_dir, "xai_80%_" + pathlib.Path(dataset_filename).stem + "_" + cam_technique)
    _create_dirs(xai_output_path)

    # Loading model
    model = load_resnet18_based_model(model_dir, device)

    # Loading data
    dataset = get_dataset_transformed(db_dir, model_dir, dataset_filename)
    X = np.array([dataset[i][0] for i in range(len(dataset))])
    y = np.array([dataset[i][1] for i in range(len(dataset))])
    input_tensor = torch.from_numpy(X)
    input_tensor.to(device)
    paths_list = [dataset.get_path(i) for i in range(len(dataset))]

    # Selecting layers where to apply CAM
    target_layers = [model.layer4[-1]]

    # Iterating over all images and targets
    for i in range(len(X)):
        for j, targets in enumerate([[ClassifierOutputTarget(0)], [ClassifierOutputTarget(1)]]):
            with cam_fun(model=model, target_layers=target_layers) as cam:
                # Generating the saliency map
                grayscale_cam = cam(input_tensor=input_tensor[i:1 + i], targets=targets)

                # Making the prediction with the model
                y_pred, _ = make_prediction(model, input_tensor[i:1 + i], device)

                # Reading and resizing the current input image
                bgr_img = cv2.imread(paths_list[i], 1)
                bgr_img = cv2.resize(bgr_img, (256, 256))

                # Computing output path depending on target class and prediction
                curr_folder = os.path.join(xai_output_path, _get_subfolder(y_pred[0], y[i]))

                # Writing input image to filesystem
                cv2.imwrite(os.path.join(curr_folder, str(i) + ".jpg"), bgr_img)

                # Writing cam merged on image to filesystem
                bgr_img = np.float32(bgr_img) / 255
                visualization = show_cam_on_image(bgr_img, grayscale_cam[0], use_rgb=False)
                cv2.imwrite(os.path.join(curr_folder, str(i) + "_target_" + str(j) + ".jpg"), visualization)


def _shap_single_sample(i, shap_values, img_numpy, xai_output_path, y_pred, y, shap_values_lim):
    """
    Compute and saves shap explanations for single sample of index i.
    :param i: index of the sample.
    :param shap_values: shap_values.
    :param img_numpy: current image as a numpy array for displaying in the plot
    :param xai_output_path: path where to save the results.
    :param y_pred: prediction of the model for the current sample.
    :param y: label of the current sample.
    :return: Non
    """

    # Transposing shap values so their shape fits with the image plotter
    shap_values = list(np.transpose(shap_values, (4, 0, 1, 2, 3)))

    # Inverting the SHAP values for class 0 so that positive shap values always mean class 1
    shap_values[0] = -shap_values[0]

    # Plotting the output that contains the explanations, and removing the legend
    shap.image_plot(shap_values=shap_values, pixel_values=img_numpy, show=False, width=18,
                    colormap_lim=max(abs(shap_values_lim[0]), abs(shap_values_lim[1])))
    # shap.image_plot(shap_values=shap_values, pixel_values=img_numpy, show=False, width=18)
    plt.gcf().axes[-1].remove()

    # Loading the plot as a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', dpi=600)
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    # Cropping the white border
    im = crop_white_border(im)

    w, h = im.size
    w_step = w / 3

    # Computing output path depending on target class and prediction
    curr_folder = os.path.join(xai_output_path, _get_subfolder(y_pred, y))

    def process_img(img):
        return add_margin(crop_white_border(img),
                          20, 20, 20, 20, (255, 255, 255, 255)).resize((700, 700))

    # Extracting the three subparts of the image plot
    original_image = process_img(im.crop((w_step * 0, 0, w_step * 1, h)))
    shap_expl_1 = process_img(im.crop((w_step * 1, 0, w_step * 2, h)))
    shap_expl_2 = process_img(im.crop((w_step * 2, 0, w_step * 3, h)))

    # Saving the images to disk
    original_image.save(os.path.join(curr_folder, str(i + 1) + ".jpg"))
    shap_expl_1.save(os.path.join(curr_folder, str(i + 1) + "target_" + "0" + ".jpg"))
    shap_expl_2.save(os.path.join(curr_folder, str(i + 1) + "target_" + "1" + ".jpg"))


def generate_shap_resnet18(db_dir, dataset_filename, model_dir, device="cuda:0", n_jobs=-1, dataset_size=None,
                           masker="blur(128,128)"):
    """
    Generating shap explanations for the given model and dataset, which are stored into the model directory.
    :param db_dir: root of the database.
    :param dataset_filename: filename of the dataset.
    :param model_dir: path of the model directory.
    :param device: device to use for pytorch computation.
    :param n_jobs: number of jobs to run in parallel.
    :param dataset_size: elements of the dataset are loaded until the size reaches this value. If None, the whole
    dataset is loaded.
    :param masker : masker to be applied on masked regions of the image (default : blurring). If "ndarray" is specified,
    a numpy array of the dimension of the image is created, with values corresponding to the white areas of the image.
    Thus, the mask applied is a white zone that actually removes symbols from the image.
    :return: None.
    """

    # Creating directories
    xai_output_path = os.path.join(model_dir, "xai_80%_" + pathlib.Path(dataset_filename).stem + "_" + "shap_" + str(masker))
    _create_dirs(xai_output_path)

    # Loading model
    model = load_resnet18_based_model(model_dir, device)

    # Loading data
    dataset = get_dataset_transformed(db_dir, model_dir, dataset_filename, max_size=dataset_size)
    X = np.array([dataset[i][0] for i in range(len(dataset))])
    y = np.array([dataset[i][1] for i in range(len(dataset))])
    input_tensor = torch.from_numpy(X)
    input_tensor.to(device)
    paths_list = [dataset.get_path(i) for i in range(len(dataset))]
    Xtr = nchw_to_nhwc(torch.from_numpy(X))

    # Making the prediction with the model
    y_pred = []
    for i in range(len(X)):
        pred, _ = make_prediction(model, input_tensor[i:i + 1], device)
        y_pred.extend(pred)

    def load_image_without_norm_add_border(image_path):
        image = Image.open(image_path)
        image = add_border(image)
        return resnet18_preprocess_no_norm(image).unsqueeze(0)

    # Loading all the original images as a numpy array.
    # Adding a black border so that the content of the image won't be cropped when shap output images are generated
    img_numpy = np.array([np.transpose(load_image_without_norm_add_border(paths_list[i]).numpy(), (0, 2, 3, 1)) for i in
                          range(len(X))]).squeeze()

    # If masking is done by ndarray, creating the masker with the same dimension as an image, filled with the value
    # corresponding to the white pixels of the images after normalization (which is done by get_dataset_transformed).
    if masker == "ndarray":
        masker = np.ones((256, 256, 3)).reshape(-1, ) * np.max(X)

    masker_f = shap.maskers.Image(masker, Xtr[0].shape)

    def predict(img: torch.Tensor) -> torch.Tensor:
        img = torch.from_numpy(img)
        img = nhwc_to_nchw(img)
        img = img.to(device)
        output = model(img)
        return output

    # Computing shap values for the whole dataset
    explainer = shap.Explainer(predict, masker=masker_f, output_names=["0", "1"])
    shap_values_list = []
    print("Computing shap values")
    # min_shap_value = float("inf")
    # max_shap_value = float("-inf")
    # for i in tqdm.tqdm(range(len(X))):
    #     shap_values_curr_sample = explainer(
    #         Xtr[i:i + 1], max_evals=200, batch_size=50
    #     )

    shap_values = explainer(
        Xtr, max_evals=100, batch_size=50
    )
    min_shap_value = np.min(shap_values.values)
    max_shap_value = np.max(shap_values.values)

    # Parallel computation of the images for the whole dataset.
    print("Generating shap images")
    Parallel(n_jobs=n_jobs)(delayed(_shap_single_sample)(
        i, shap_values.values[i: i + 1], img_numpy[i: i + 1],
        xai_output_path, y_pred[i], y[i], (min_shap_value, max_shap_value)) for i in tqdm.tqdm(range(len(X))))
