import csv
import json
import os
import pathlib
import shutil
import tempfile
from os.path import join

import cv2
import shap
from pytorch_grad_cam import GradCAM
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from xaipatimg.datagen.dbimg import load_db
from xaipatimg.datagen.genimg import gen_img
from xaipatimg.datagen.utils import get_coords_diff, PatImgObj, random_mutation
from xaipatimg.ml import resnet18_preprocess_no_norm
from xaipatimg.ml._colors import red_transparent_green
from xaipatimg.ml.learning import get_dataset_transformed, make_prediction, load_resnet_based_model
import numpy as np
import io
from xaipatimg.ml.utils import nhwc_to_nchw, nchw_to_nhwc, add_margin, crop_white_border, add_border, \
    generate_llm_text_image
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


def _predict(model_dir, device, dataset, resnet_type):
    """
    Utility function to make predictions with the given model on the given dataset
    :param model_dir: path of the model directory.
    :param device: device to use for predictions.
    :param dataset: dataset as a PatImgDataset instance.
    :return: The input X and y matrices, the predicted y_pred vector and the model : X, y, y_pred, model
    """
    model = load_resnet_based_model(model_dir, device, resnet_type=resnet_type)
    X = np.array([dataset[i][0] for i in range(len(dataset))])
    y = np.array([dataset[i][1] for i in range(len(dataset))])
    input_tensor = torch.from_numpy(X)
    input_tensor.to(device)

    # Making the prediction with the model
    y_pred = []
    for i in range(len(X)):
        pred, _ = make_prediction(model, input_tensor[i:i + 1], device)
        y_pred.extend(pred)

    return X, y, y_pred, model


def _vertical_concatenation(top_img, bottom_img):
    """
    Vertical concatenation of two images. The final image keeps the width of the top image. The height of the bottom
    image might be adjusted so that the ratios are kept.

    :param top_img: top image
    :param bottom_img: bottom image
    :return: contatenated image
    """

    # Get the target width
    target_width = top_img.width

    # Compute the proportional height
    w_percent = target_width / bottom_img.width
    target_height = int(bottom_img.height * w_percent)

    # Resize with high-quality resampling
    bottom_img = bottom_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Determine final canvas size
    final_width = max(top_img.width, bottom_img.width)
    final_height = top_img.height + bottom_img.height

    # Create a new blank image with appropriate mode
    combined = Image.new(top_img.mode, (final_width, final_height), (255, 255, 255))

    # Compute horizontal offsets to center images if widths differ
    x1 = (final_width - top_img.width) // 2
    x2 = (final_width - bottom_img.width) // 2

    # Paste images
    combined.paste(top_img, (x1, 0))
    combined.paste(bottom_img, (x2, int(top_img.height - 0.2 * bottom_img.height)))
    return combined


def _generate_displayable_explanation(model_pred, explanation_img, yes_pred_img_path, no_pred_img_path, out_path,
                                      output_size, left_ratio=0.5, font_size=40, padding=20, AI_only=False):
    """
    Generating the image that can be displayed to users and that contains both the prediction of the model and the
    generated explanation.
    Left size : model prediction. Right size : explanation.
    If AI_only is set to True, only the model prediction will be displayed.

    :param model_pred: model prediction.
    :param explanation_img: explanation image.
    :param yes_pred_img_path: Image which represents a "yes" prediction.
    :param no_pred_img_path: Image which represents a "no" prediction.
    :param out_path: path where the image is saved.
    :param output_size: size of the output image in pixels.
    :param left_ratio: proportion of the image to be displayed on the left side.
    :param font_path: path to the font to be used.
    :param font_size: font size.
    :param padding: padding.
    :param AI_only: If True, only the model prediction will be displayed.
    :return:
    """

    total_width, total_height = output_size

    # Load prediction image
    pred_img = Image.open(yes_pred_img_path if model_pred == 1 else no_pred_img_path).convert("RGBA")

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
    except IOError:
        print("Warning: LiberationSans-Regular.ttf not found. Falling back to default font.")
        font = ImageFont.load_default()

    left_title = "AI prediction"
    right_title = "AI justification"

    # Measure text
    tmp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left_text_w = tmp_draw.textlength(left_title, font=font)
    left_text_h = font.getbbox(left_title)[3] - font.getbbox(left_title)[1]
    if not AI_only:
        right_text_w = tmp_draw.textlength(right_title, font=font)
        right_text_h = font.getbbox(right_title)[3] - font.getbbox(right_title)[1]

    def scale_to_fit(img, max_w, max_h):
        w, h = img.size
        scale = min(max_w / w, max_h / h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    canvas = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if AI_only:
        # --- Single centered panel, but keep left_ratio width ---
        panel_width = int(total_width * left_ratio)
        panel_x = (total_width - panel_width) // 2  # horizontally center the panel

        max_img_height = total_height - (left_text_h + 3 * padding)
        pred_img_resized = scale_to_fit(pred_img,
                                        panel_width - 2 * padding,
                                        max_img_height)

        block_height = left_text_h + padding + pred_img_resized.height
        top_block = (total_height - block_height) // 2

        text_x = panel_x + padding + (panel_width - left_text_w) // 2
        draw.text((text_x, top_block), left_title, fill="black", font=font)

        canvas.paste(pred_img_resized,
                     (panel_x + padding + (panel_width - pred_img_resized.width) // 2,
                      top_block + left_text_h + padding),
                     pred_img_resized)

    else:
        # --- Two panels ---
        left_panel_width = int(total_width * left_ratio)
        right_panel_width = total_width - left_panel_width

        max_left_img_height = total_height - (left_text_h + 3 * padding)
        pred_img_resized = scale_to_fit(pred_img,
                                        left_panel_width - 2 * padding,
                                        max_left_img_height)

        block_height_left = left_text_h + padding + pred_img_resized.height
        top_left_block = (total_height - block_height_left) // 2
        left_text_x = padding + (left_panel_width - left_text_w) // 2
        draw.text((left_text_x, top_left_block), left_title, fill="black", font=font)
        canvas.paste(pred_img_resized,
                     (padding + (left_panel_width - pred_img_resized.width) // 2,
                      top_left_block + left_text_h + padding),
                     pred_img_resized)

        # Right panel
        max_right_img_height = total_height - (right_text_h + 3 * padding)
        justif_img_resized = scale_to_fit(explanation_img.convert("RGBA"),
                                          right_panel_width - 2 * padding,
                                          max_right_img_height)

        right_offset_x = left_panel_width
        right_text_x = right_offset_x + padding + (right_panel_width - right_text_w) // 2
        draw.text((right_text_x, padding), right_title, fill="black", font=font)
        canvas.paste(justif_img_resized,
                     (right_offset_x + padding + (right_panel_width - justif_img_resized.width) // 2,
                      padding + right_text_h + padding),
                     justif_img_resized)

    canvas.save(out_path)


def generate_cam_resnet18(cam_technique, db_dir, datasets_dir_path, dataset_filename, model_dir, device="cuda:0"):
    """
    Generating cam images for given model on test set, which are saved into the model directory.
    :param cam_technique: description of the cam technique (str) to use to generate explanations
    :param db_dir: root of the database.
    :param datasets_dir_path: path where the datasets are located
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
    xai_output_path = os.path.join(model_dir, "xai_" + pathlib.Path(dataset_filename).stem + "_" + cam_technique)
    _create_dirs(xai_output_path)

    # Loading data
    dataset = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, dataset_filename)

    # Make prediction
    X, y, _, model = _predict(model_dir, device, dataset)

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

                # Writing input image to filesystem
                cv2.imwrite(os.path.join(xai_output_path, str(i) + ".jpg"), bgr_img)

                # Writing cam merged on image to filesystem
                bgr_img = np.float32(bgr_img) / 255
                visualization = show_cam_on_image(bgr_img, grayscale_cam[0], use_rgb=False)
                cv2.imwrite(os.path.join(xai_output_path, str(i) + "_target_" + str(j) + ".jpg"), visualization)


def _shap_single_sample(sample_idx, shap_values, img_numpy, xai_output_path, y_pred, shap_values_lim,
                        shap_scale_img_path, yes_pred_img_path, no_pred_img_path):
    """
    Compute and saves shap explanations for single sample of index i.
    :param sample_idx: index of the sample.
    :param shap_values: shap_values.
    :param img_numpy: current image as a numpy array for displaying in the plot
    :param xai_output_path: path where to save the results.
    :param y_pred: prediction of the model for the current sample.
    :param shap_scale_img_path : path to the image that represents the shap color scale and which will be added to the
    bottom of the generated explanation. Will be ignored if None.
    :param yes_pred_img_path: path to the image that represent a yes prediction
    :param no_pred_img_path: path to the image that represent a no prediction
    :return: Non
    """

    # Transposing shap values so their shape fits with the image plotter
    shap_values = list(np.transpose(shap_values, (4, 0, 1, 2, 3)))

    # Inverting the SHAP values for class 0 so that positive shap values always mean class 1
    shap_values[0] = -shap_values[0]

    # Plotting the output that contains the explanations, and removing the legend
    shap.image_plot(shap_values=shap_values, pixel_values=img_numpy, show=False, width=18,
                    colormap_lim=max(abs(shap_values_lim[0]), abs(shap_values_lim[1])),
                    cmap=red_transparent_green)

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

    def process_img(img):
        return add_margin(crop_white_border(img),
                          20, 20, 20, 20, (255, 255, 255, 255)).resize((700, 700))

    # Extracting the three subparts of the image plot
    original_image = process_img(im.crop((w_step * 0, 0, w_step * 1, h)))
    shap_expl_1 = process_img(im.crop((w_step * 1, 0, w_step * 2, h)))

    # Adding the color scale to the SHAP explanation image
    if shap_scale_img_path is not None:
        scale_img = Image.open(shap_scale_img_path)
        shap_expl_1 = _vertical_concatenation(shap_expl_1, scale_img)

    # Saving the images to disk
    original_image.save(os.path.join(xai_output_path, str(sample_idx) + "_og.jpg"))

    output_img_path = os.path.join(xai_output_path, str(sample_idx) + ".png")
    _generate_displayable_explanation(y_pred, shap_expl_1, yes_pred_img_path, no_pred_img_path, output_img_path,
                                      output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5)

    output_img_AIonly_path = os.path.join(xai_output_path, str(sample_idx) + "AIonly.png")
    _generate_displayable_explanation(y_pred, None, yes_pred_img_path, no_pred_img_path, output_img_AIonly_path,
                                      output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5, AI_only=True)


def generate_shap_resnet(db_dir, datasets_dir_path, dataset_filename, model_dir, xai_output_path, yes_pred_img_path,
                           no_pred_img_path, device="cuda:0", n_jobs=-1, dataset_size=None, masker="blur(128,128)",
                           shap_scale_img_path=None, max_evals=10000, resnet_type="resnet18"):
    """
    Generating shap explanations for the given model and dataset, which are stored into the model directory.
    :param db_dir: root of the database.
    :param datasets_dir_path: path to the directory where the datasets are located.
    :param dataset_filename: filename of the dataset.
    :param model_dir: path of the model directory.
    :param xai_output_path: path where to save the results.
    :param device: device to use for pytorch computation.
    :param n_jobs: number of jobs to run in parallel.
    :param dataset_size: elements of the dataset are loaded until the size reaches this value. If None, the whole
    dataset is loaded.
    :param masker : masker to be applied on masked regions of the image (default : blurring). If "ndarray" is specified,
    a numpy array of the dimension of the image is created, with values corresponding to the white areas of the image.
    Thus, the mask applied is a white zone that actually removes symbols from the image.
    :param shap_scale_img_path : path to the image that represents the shap color scale and which will be added to the
    bottom of the generated explanation. Will be ignored if None.
    :param yes_pred_img_path: path to the image that represents the yes prediction.
    :param no_pred_img_path: path to the image that represents the no prediction.
    :param max_evals: maximum number of evaluation runs to run.
    :param resnet_type: type of resnet model to use (either "resnet18" or "resnet50").
    :return: None.
    """

    # Creating directories
    _create_dirs(xai_output_path)

    # Loading data
    dataset = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, dataset_filename, max_size=dataset_size)

    # Make prediction
    X, _, y_pred, model = _predict(model_dir, device, dataset, resnet_type=resnet_type)

    Xtr = nchw_to_nhwc(torch.from_numpy(X))

    def load_image_without_norm_add_border(image_path):
        image = Image.open(image_path)
        image = add_border(image)
        return resnet18_preprocess_no_norm(image).unsqueeze(0)

    # Loading all the original images as a numpy array.
    # Adding a black border so that the content of the image won't be cropped when shap output images are generated
    paths_list = [dataset.get_path(i) for i in range(len(dataset))]
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
    print("Computing shap values")
    explainer = shap.Explainer(predict, masker=masker_f, output_names=["0", "1"])
    shap_values = explainer(
        Xtr, max_evals=max_evals, batch_size=50
    )
    min_shap_value = np.min(shap_values.values)
    max_shap_value = np.max(shap_values.values)

    # Parallel computation of the images for the whole dataset.
    print("Generating shap images")
    Parallel(n_jobs=n_jobs)(delayed(_shap_single_sample)(
        i, shap_values.values[i: i + 1], img_numpy[i: i + 1], xai_output_path, y_pred[i],
        (min_shap_value, max_shap_value), shap_scale_img_path,
        yes_pred_img_path, no_pred_img_path) for i in tqdm.tqdm(range(len(X))))


def generate_counterfactuals_resnet_random_approach(db_dir, datasets_dir_path, dataset_filename, model_dir,
                                                      xai_output_path, yes_pred_img_path, no_pred_img_path, shapes,
                                                      colors, empty_probability, max_depth, nb_tries_per_depth,
                                                      generic_rule_fun, devices, n_jobs=-1, dataset_size=None,
                                                      pos_pred_legend_path=None, neg_pred_legend_path=None,
                                                      resnet_type="resnet18", **kwargs):
    """
    Generating counterfactual explanations for the given model and dataset. Explanations are obtained by assessing
    randomly mutated images. A mutation consists in randomly changing the content of a cell of the image.
    The algorithm used is the following:

    for curr_depth in (1..max_depth):
        for curr_try in (1..nb_tries_per_depth):
            cf_try <- random_mutation(depth=curr_depth)
            if cf_try is a counterfactual:
                return cf_try

    :param db_dir: root of the database.
    :param datasets_dir_path: path to the directory where the datasets are located.
    :param dataset_filename: filename of the dataset.
    :param model_dir: path of the model directory.
    :param xai_output_path: path where to save the results.
    :param yes_pred_img_path: path to the image that represents the yes prediction.
    :param no_pred_img_path: path to the image that represents the no prediction.
    :param generic_rule_fun: generic rule function that verifies if any sample respects the rule or not.
    :param devices: List of devices to use for pytorch computation. Jobs are distributed to all devices.
    :param n_jobs: number of jobs to run in parallel.
    :param dataset_size: elements of the dataset are loaded until the size reaches this value. If None, the whole
    dataset is loaded.
    :param max_depth: maximum number of random mutations in the generated counterfactuals.
    :param nb_tries_per_depth: number of mutations which are assessed for each depth.
    :param shapes: list of possible shapes.
    :param colors: list of possible colors.
    :param empty_probability: probability of an empty cell.
    :param pos_pred_legend_path: path to the legend when the prediction is positive.
    :param neg_pred_legend_path: path to the legend when the prediction is negative.
    :param resnet_type: 'resnet18' or 'resnet50'.
    :return: None.
    """

    # Creating directories
    _create_dirs(xai_output_path)

    # Loading data
    dataset = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, dataset_filename, max_size=dataset_size)

    # Make prediction
    X, y, y_pred, model = _predict(model_dir, devices[0], dataset, resnet_type=resnet_type)

    # Load database
    db = load_db(db_dir)

    # Parallel computation of the images for the whole dataset.
    print("Generating counterfactual images")
    return_values = Parallel(n_jobs=n_jobs)(delayed(_cf_single_sample_random_approach)(
        db_dir,  # db_dir
        datasets_dir_path,  # datasets_dir_path
        i,  # sample_idx
        xai_output_path,  # xai_output_path
        db[dataset.get_id(i)],  # img_entry
        y[i],  # y
        y_pred[i],  # y_pred
        model_dir,  # model_dir
        devices[i % len(devices)],  # device
        max_depth,  # max_depth
        nb_tries_per_depth,  # nb_tries_per_depth
        shapes,  # shapes
        colors,  # colors
        empty_probability,  # empty_probability
        generic_rule_fun,  # generic_rule_fun
        yes_pred_img_path,  # yes_pred_img_path
        no_pred_img_path,  # no_pred_img_path
        pos_pred_legend_path,  # pos_pred_legend_path
        neg_pred_legend_path,  # neg_pred_legend_path
        resnet_type, # resnet_type
        **kwargs) for i in tqdm.tqdm(range(len(X))))

    cf_depths, cf_verified_depths = zip(*return_values)

    # Writing depths of generated counterfactuals
    csv_data = [
        {"CF_id": str(i), "cf_model_depth": str(cf_depths[i]), "cf_function_depth": str(cf_verified_depths[i])}
        for i in range(len(X))
    ]
    with open(os.path.join(xai_output_path, "CF_generation.csv"), 'w', newline='') as csvfile:
        fieldnames = ['CF_id', 'cf_model_depth', 'cf_function_depth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)


def _cf_single_sample_random_approach(db_dir, datasets_dir_path, sample_idx, xai_output_path, img_entry, y, y_pred,
                                      model_dir, device, max_depth, nb_tries_per_depth, shapes, colors,
                                      empty_probability, generic_rule_fun, yes_pred_img_path, no_pred_img_path,
                                      pos_pred_legend_path=None,
                                      neg_pred_legend_path=None, resnet_type="resnet18", **kwargs):
    """
    Generating a set of counterfactual explanations for a single sample with the random approach.
    Generating a counterfactual to the model as well as a counterfactual to the actual sample according to the generic
    rule function.
    :param db_dir: path of the database.
    :param sample_idx: index of the sample.
    :param xai_output_path: path where to save the xai explanations.
    :param img_entry: entry of the image in the database.
    :param y : actual class of the sample
    :param y_pred: predicted class of the sample.
    :param model_dir: path of the model directory.
    :param device: device to use for pytorch computation.
    :param max_depth: maximum number of random mutations in the generated counterfactuals.
    :param nb_tries_per_depth: number of mutations which are assessed for each depth.
    :param shapes: list of possible shapes.
    :param colors: list of possible colors.
    :param empty_probability: probability of an empty cell.
    :param generic_rule_fun: generic rule function that verifies if any sample respects the rule or not.
    :param yes_pred_img_path: path to the image that represents the yes prediction.
    :param no_pred_img_path: path to the image that represents the no prediction.
    :param pos_pred_legend_path: path to the legend when the prediction is positive.
    :param neg_pred_legend_path: path to the legend when the prediction is negative.
    :param resnet_type: 'resnet18' or 'resnet50'.
    :return: None
    """

    model_cf_found_dict = None
    model_cf_found_depth = None
    rule_cf_found_dict = None
    rule_cf_found_depth = None

    for curr_depth in range(1, max_depth + 1):

        mutations_dict_list = []

        # Generate possible counterfactuals of given mutation depth
        for _ in range(nb_tries_per_depth):
            mutations_dict_list.append(random_mutation(img_entry, curr_depth, shapes, colors, empty_probability))

        # Create the temporary directory and store the path
        temp_dir = tempfile.mkdtemp()

        # Saving the original image
        og_img_path = join(temp_dir, "original.png")
        gen_img(og_img_path, img_entry["cnt"], img_entry["div"], img_entry["size"])

        possible_cf_paths = []
        # Writing the possible counterfactuals into the temporary directory
        for cf_id, cf_dict in enumerate(mutations_dict_list):
            curr_cf_path = join(temp_dir, str(cf_id) + ".png")
            possible_cf_paths.append(curr_cf_path)
            gen_img(curr_cf_path, cf_dict["cnt"], cf_dict["div"], cf_dict["size"])

        # Writing a CSV dataset of counterfactuals to be able to pass them through the model
        dataset_path = os.path.join(temp_dir, "dataset.csv")
        y_cf = [None for _ in range(len(possible_cf_paths))]  # These values will not be used so they are set to None
        csv_content_train = np.array([np.concatenate((["path"], possible_cf_paths), axis=0),
                                      np.concatenate((["class"], y_cf), axis=0)]).T
        with open(dataset_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_content_train)

        # Verifying that the generated samples are counterfactuals to the model or to the rule function
        dataset = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, dataset_path)
        _, _, y_pred_model_cf, _ = _predict(model_dir, device, dataset, resnet_type=resnet_type)
        for y_pred_model_cf_idx, y_pred_curr_cf in enumerate(y_pred_model_cf):

            # Verifying that the generated sample is an actual counterfactual to the model (exclusive or)
            if model_cf_found_dict is None and bool(y_pred) ^ bool(y_pred_curr_cf):
                model_cf_found_dict = mutations_dict_list[y_pred_model_cf_idx]
                model_cf_found_depth = curr_depth

            # Verifying that the generated sample is an actual counterfactual to generic rule function (differs from the
            # model counterfactual if the model makes the wrong prediction for the sample).
            if rule_cf_found_dict is None:
                gen_fun_return, _ = generic_rule_fun(mutations_dict_list[y_pred_model_cf_idx]["cnt"], **kwargs)
                y_cf = int(gen_fun_return)
                if y_cf != y:
                    rule_cf_found_dict = mutations_dict_list[y_pred_model_cf_idx]
                    rule_cf_found_depth = curr_depth

            if model_cf_found_dict is not None and rule_cf_found_dict is not None:
                break

        # Generating the output for counterfactuals, if both were found
        if model_cf_found_dict is not None and rule_cf_found_dict is not None:

            # Copying the actual original image into the XAI output directory
            shutil.copyfile(og_img_path, os.path.join(xai_output_path, str(sample_idx) + "_og.png"))

            coords_diff = get_coords_diff(PatImgObj(img_entry), PatImgObj(model_cf_found_dict))
            cf_img = gen_img(None, model_cf_found_dict["cnt"], model_cf_found_dict["div"],
                             model_cf_found_dict["size"],
                             to_highlight=coords_diff, return_image=True)

            # Adding the legend to the explanation
            if pos_pred_legend_path is not None and neg_pred_legend_path is not None:
                legend_img = Image.open(pos_pred_legend_path) if y_pred == 1 else Image.open(neg_pred_legend_path)
                cf_img = _vertical_concatenation(cf_img, legend_img)

            # Saving the images to disk
            output_img_path = os.path.join(xai_output_path, str(sample_idx) + ".png")
            _generate_displayable_explanation(y_pred, cf_img, yes_pred_img_path, no_pred_img_path, output_img_path,
                                              output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5)

            output_img_AIonly_path = os.path.join(xai_output_path, str(sample_idx) + "AIonly.png")
            _generate_displayable_explanation(y_pred, None, yes_pred_img_path, no_pred_img_path,
                                              output_img_AIonly_path,
                                              output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5,
                                              AI_only=True)

            # Exporting the content of the JSON explanation for the model counterfactual and function counterfactual
            with open(os.path.join(xai_output_path, f"{sample_idx}.json"), 'w', encoding='utf-8') as f:
                json.dump(model_cf_found_dict, f, ensure_ascii=False, indent=4)
            with open(os.path.join(xai_output_path, f"{sample_idx}_function.json"), 'w', encoding='utf-8') as f:
                json.dump(rule_cf_found_dict, f, ensure_ascii=False, indent=4)

            # Returning the depths that were needed to obtain the counterfactual explanations
            return model_cf_found_depth, rule_cf_found_depth

        # Removing the temporary dictionary
        shutil.rmtree(temp_dir)

    # In case no counterfactual was found at any depth, returning None
    return None, None

def _llm_generate_single_explanation(db, llm_model, question, explicit_colors_dict, tokenizer, dataset, y, y_pred, idx,
                                     path_to_counterfactuals_dir_for_model_errors, pos_llm_scaffold, neg_llm_scaffold,
                                     division_X, division_Y, pattern_dict=None):
    """
    Generating the LLM explanation
    :param db: database dictionary.
    :param llm_model: instanciated LLM model.
    :param tokenizer: instanciated tokenizer.
    :param dataset: PatImgDataset instance.
    :param y: true label of the sample.
    :param y_pred: prediction of the model for the sample.
    :param idx: index of explanation to generate.
    :param path_to_counterfactuals_dir_for_model_errors: path to the directory in which counterfactual explanations have
    been generated. If not None, the counterfactual explanation is used instead of the actual sample to generate the
    LLM explanation when the model makes the wrong prediction. This ensures that the generated explanation is consistent
    with the model prediction.
    :param pos_llm_scaffold: scaffold of an explanation the LLM is supposed to generate for a positive instance.
    :param neg_llm_scaffold: scaffold of an explanation the LLM is supposed to generate for a negative instance.
    :param division_X: number of X divisions
    :param division_Y: number of Y divisions
    :param pattern_dict: if the task is to search for a pattern of symbols, JSON description of the pattern.
    :return:
    """

    def custom_sort(a):
        """
        Custom function to sort dict elements row first then columns
        :param a:
        :return:
        """
        return a["pos"][1], a["pos"][0]

    # Function that converts the dict image content to an equivalent with explicit color names and using the
    # (A..F)(1..6) coordinates notation
    def convert_content(img_content, explicit_colors_dict, for_pattern=False):

        # Create dictionaries associating x and y coordinates to the corresponding value in the (A..X)(1..N) coordinates
        # notation
        dict_coords_y = {i: str(1 + i) for i in range(division_Y)}
        dict_coords_x = {i: chr(ord("A") + i) for i in range(division_X)}

        if for_pattern:
            dict_coords_y = {i: f"N" if i == 0 else f"N+{i}" for i in range(division_Y)}
            dict_coords_x = {i: f"X" if i == 0 else f"X+{i}" for i in range(division_X)}
        else:
            dict_coords_y = {i: str(1 + i) for i in range(division_Y)}
            dict_coords_x = {i: chr(ord("A") + i) for i in range(division_X)}

        new_content = []
        for shape_content in img_content:
            new_shape_content = {"shp": shape_content["shp"]}
            x_pos, y_pos = shape_content["pos"][0], shape_content["pos"][1]
            new_coord = dict_coords_x[x_pos] + dict_coords_y[y_pos]
            new_shape_content["pos"] = new_coord
            new_shape_content["col"] = explicit_colors_dict[shape_content["col"]]
            new_content.append(new_shape_content)

        # Sorting the elements row-first
        # new_content = sorted(new_content, key=custom_sort)

        return new_content

    pattern_dict_converted = convert_content(pattern_dict, explicit_colors_dict, for_pattern=True) if pattern_dict is not None else None

    system_prompt = (f"You are the explainability system of an AI model. Your role is to justify the decisions of "
                     f"the model. The role of the model is to answer questions about the content of images of "
                     f"symbols of colors. The images are described in a JSON data structure. The coordinates "
                     f"system uses letters from A to F for the columns and numbers from 1 to 6 for the rows. The "
                     f"user will provide you the prediction of the AI model for a given image and the corresponding "
                     f"JSON data. You need to give an explanation of the prediction. The explanation is expected to be "
                     f"a very short sentence which introduces a list of all coordinates that are involved in the model's"
                     f" prediction. The justification sentence and the list of coordinates must be separated by the "
                     f"character '|'. The coordinates are separated with the symbol ';', and there is no need to sort them. "
                     f"Do not use escape characters or markdown syntax. The question the model must answer "
                     f"is '{question}'.{f"The pattern to search for is "
                                        f"{pattern_dict_converted}. Here X correspond to any letter coordinate, and N"
                                        f"to any number coordinate." if pattern_dict is not None else ""} "
                     f"Here are examples of justifications for a positive and a negative sample. "
                     f"Positive : '{pos_llm_scaffold}' Negative : '{neg_llm_scaffold}'")

    print(system_prompt)

    # Loading the rule's counterfactual explanation instead of the original sample in case the model makes the wrong
    # prediction, so that the class of the sample matches the class predicted by the model.
    if path_to_counterfactuals_dir_for_model_errors is not None and y != y_pred:
        with open(os.path.join(path_to_counterfactuals_dir_for_model_errors, f"{idx}_function.json")) as json_data:
            img_content = str(convert_content(json.load(json_data)["cnt"], explicit_colors_dict))
    # Otherwise loading the image content of the original sample
    else:
        img_id = dataset.get_id(idx)
        img_content = str(convert_content(db[img_id]["cnt"], explicit_colors_dict))

    user_prompt = f"The AI model predicts {"Yes" if y_pred == 1 else "No"} for the image : {img_content}"
    messages = [
        {"role": "system", "cnt": system_prompt},
        {"role": "user", "cnt": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(llm_model.device)

    # Performing LLM computation
    generated = llm_model.generate(**inputs, max_new_tokens=5000)

    # Parsing LLM answer
    full_answer = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:])
    print(full_answer)
    parsed_answer = full_answer.partition("<|start|>assistant<|channel|>final<|message|>")[2]
    parsed_answer = parsed_answer.partition("<|return|>")[0]
    return parsed_answer

def generate_LLM_explanations(db_dir, db, datasets_dir_path, dataset_filename, model_dir, llm_model, llm_tokenizer,
                              xai_output_path, explicit_colors_dict, question, yes_pred_img_path, no_pred_img_path,
                              yes_pred_img_path_small, no_pred_img_path_small, pos_llm_scaffold, neg_llm_scaffold,
                              X_division, Y_division, pattern_dict=None, device="cuda:0", dataset_size=None,
                              only_for_index=None, path_to_counterfactuals_dir_for_model_errors=None,
                              resnet_type="resnet18"):
    """
    Generating LLM explanations for the given model and dataset.
    :param db_dir: root of the database.
    :param db: database object.
    :param datasets_dir_path: path to the directory where the datasets are located.
    :param dataset_filename: filename of the dataset.
    :param model_dir: path of the model directory.
    :param llm_model: llm model to use for inference.
    :param llm_tokenizer: llm tokenizer.
    :param xai_output_path: path where to save the results.
    :param explicit_colors_dict: dictionary that associates color hex codes with their explicit name.
    :param question: question corresponding to the given model.
    :param yes_pred_img_path: path to the image that represents the yes prediction.
    :param no_pred_img_path: path to the image that represents the no prediction.
    :param yes_pred_img_path: path to the image that represents the yes for small rendering (as part of the LLM explanation).
    :param no_pred_img_path: path to the image that represents the no for small rendering (as part of the LLM explanation).
    :param pos_llm_scaffold: scaffold of an explanation the LLM is supposed to generate for a positive instance.
    :param neg_llm_scaffold: scaffold of an explanation the LLM is supposed to generate for a negative instance.
    :param X_division: number of X divisions.
    :param Y_division: number of Y divisions.
    :param pattern_dict: if the task is to search for a pattern of symbols, JSON description of the pattern.
    :param device: device to use for pytorch computation. The LLM will use all GPUs available.
    :param dataset_size: elements of the dataset are loaded until the size reaches this value. If None, the whole
    dataset is loaded.
    :param only_for_index: If not None, only the given indexes will be considered.
    :param path_to_counterfactuals_dir_for_model_errors: If not None, when the model makes the wrong prediction, the
    counterfactual explanation generated for the sample will be used instead of the actual sample to generate the
    LLM explanation. This allows generating an explanation which is consistent with the model prediction.
    :param resnet_type: type of resnet model to use (either "resnet18" or "resnet50").
    :return:
    """
    # Creating directories
    _create_dirs(xai_output_path)
    print(xai_output_path)

    # Loading data
    dataset = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, dataset_filename, max_size=dataset_size)

    # Make prediction
    X, y, y_pred, _ = _predict(model_dir, device, dataset, resnet_type=resnet_type)

    index_list = range(len(X)) if only_for_index is None else only_for_index
    for sample_idx in tqdm.tqdm(index_list):
        expl_str = _llm_generate_single_explanation(db, llm_model, question, explicit_colors_dict, llm_tokenizer,
                                                    dataset, y[sample_idx], y_pred[sample_idx], sample_idx,
                                                    path_to_counterfactuals_dir_for_model_errors, pos_llm_scaffold,
                                                    neg_llm_scaffold, division_X=X_division, division_Y=Y_division,
                                                    pattern_dict=pattern_dict)

        expl_img = generate_llm_text_image(text=expl_str, width=700, height=700, font_size=36, line_spacing=15,
                                           pred_img_scale=1.8, yes_pred_img_path=yes_pred_img_path_small,
                                           no_pred_img_path=no_pred_img_path_small, margin=50)

        # Saving the images to disk
        output_img_path = os.path.join(xai_output_path, str(sample_idx) + ".png")
        _generate_displayable_explanation(y_pred[sample_idx], expl_img, yes_pred_img_path, no_pred_img_path,
                                          output_img_path,
                                          output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5)

        output_img_AIonly_path = os.path.join(xai_output_path, str(sample_idx) + "AIonly.png")
        _generate_displayable_explanation(y_pred[sample_idx], None, yes_pred_img_path, no_pred_img_path,
                                          output_img_AIonly_path,
                                          output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5,
                                          AI_only=True)


def create_xai_index(db_dir, datasets_dir_path, dataset_filename, model_dir, xai_dirs, dataset_size, device,
                     resnet_type="resnet18"):
    """
    Creating a csv file which contains the information of (dataset index, class, model prediction, path to image, [path to every xai generated image],
    path to the generated image for the prediction only).
    :param db_dir: root of the database.
    :param datasets_dir_path: path to the directory where the datasets are located.
    :param dataset_filename: filename of the dataset.
    :param model_dir: path of the model directory.
    :param xai_dirs: dictionary that contains for every explanation type, the path to where the explanations are saved.
    :param dataset_size: elements of the dataset are loaded until the size reaches this value. If None, the whole
    dataset is loaded.
    :param device: device to use for pytorch computation.
    :param resnet_type: type of resnet model to use (either "resnet18" or "resnet50").
    :return:
    """

    # Loading data
    dataset = get_dataset_transformed(db_dir, datasets_dir_path, model_dir, dataset_filename, max_size=dataset_size)

    # Make prediction
    X, y, y_pred, model = _predict(model_dir, device, dataset, resnet_type=resnet_type)

    csv_data = []

    for idx in range(len(dataset)):
        curr_line = {"idx": idx, "y": y[idx], "y_pred": y_pred[idx], "img": dataset.get_image_name(idx)}

        for xai_key, xai_dir in xai_dirs.items():
            curr_line[xai_key] = os.path.join(xai_dir, f"{idx}.png")
            curr_line["AI"] = os.path.join(xai_dir, f"{idx}AIonly.png")

        csv_data.append(curr_line)

    with open(os.path.join(model_dir, "xai_index.csv"), 'w', newline='') as csvfile:
        fieldnames = ["idx", "y", "y_pred", "img", "AI"]
        fieldnames.extend([k for k in xai_dirs.keys()])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
