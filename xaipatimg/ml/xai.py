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

from xaipatimg.datagen.dbimg import load_db
from xaipatimg.datagen.genimg import gen_img
from xaipatimg.datagen.utils import get_coords_diff, PatImgObj, random_mutation
from xaipatimg.ml import resnet18_preprocess_no_norm
from xaipatimg.ml._colors import red_transparent_green
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

def _predict(model_dir, device, dataset):
    """
    Utility function to make predictions with the given model on the given dataset
    :param model_dir: path of the model directory.
    :param device: device to use for predictions.
    :param dataset: dataset as a PatImgDataset instance.
    :return: The input X and y matrices, the predicted y_pred vector and the model : X, y, y_pred, model
    """
    model = load_resnet18_based_model(model_dir, device)
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
    xai_output_path = os.path.join(model_dir, "xai_" + pathlib.Path(dataset_filename).stem + "_" + cam_technique)
    _create_dirs(xai_output_path)

    # Loading data
    dataset = get_dataset_transformed(db_dir, model_dir, dataset_filename)

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
    _generate_displayable_explanation(y_pred, shap_expl_1, yes_pred_img_path, no_pred_img_path, output_img_AIonly_path,
                                      output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5, AI_only=True)


def generate_shap_resnet18(db_dir, dataset_filename, model_dir, xai_output_path, yes_pred_img_path, no_pred_img_path,
                           device="cuda:0", n_jobs=-1, dataset_size=None, masker="blur(128,128)", shap_scale_img_path=None):
    """
    Generating shap explanations for the given model and dataset, which are stored into the model directory.
    :param db_dir: root of the database.
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
    :param yes_pred_img_path: path the image that represents the yes prediction.
    :param no_pred_img_path: path the image that represents the no prediction.

    :return: None.
    """

    # Creating directories
    _create_dirs(xai_output_path)

    # Loading data
    dataset = get_dataset_transformed(db_dir, model_dir, dataset_filename, max_size=dataset_size)

    # Make prediction
    X, _, y_pred, model = _predict(model_dir, device, dataset)

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
        Xtr, max_evals=10000, batch_size=50
    )
    min_shap_value = np.min(shap_values.values)
    max_shap_value = np.max(shap_values.values)

    # Parallel computation of the images for the whole dataset.
    print("Generating shap images")
    Parallel(n_jobs=n_jobs)(delayed(_shap_single_sample)(
        i, shap_values.values[i: i + 1], img_numpy[i: i + 1], xai_output_path, y_pred[i],
        (min_shap_value, max_shap_value), shap_scale_img_path,
        yes_pred_img_path, no_pred_img_path) for i in tqdm.tqdm(range(len(X))))

def _cf_single_sample_random_approach(db_dir, sample_idx, xai_output_path, img_entry, y_pred, model_dir, device,
                                      max_depth, nb_tries_per_depth, shapes, colors, empty_probability,
                                      yes_pred_img_path, no_pred_img_path, pos_pred_legend_path=None,
                                      neg_pred_legend_path=None):
    """
    Generating a set of counterfactual explanations for a single sample with the random approach.
    :param db_dir: path of the database.
    :param sample_idx: index of the sample.
    :param xai_output_path: path where to save the xai explanations.
    :param img_entry: entry of the image in the database.
    :param y_pred: predicted class of the sample.
    :param model_dir: path of the model directory.
    :param device: device to use for pytorch computation.
    :param max_depth: maximum number of random mutations in the generated counterfactuals.
    :param nb_tries_per_depth: number of mutations which are assessed for each depth.
    :param shapes: list of possible shapes.
    :param colors: list of possible colors.
    :param empty_probability: probability of an empty cell.
    :param yes_pred_img_path: path the image that represents the yes prediction.
    :param no_pred_img_path: path the image that represents the no prediction.
    :param pos_pred_legend_path: path the legend when the prediction is positive.
    :param neg_pred_legend_path: path the legend when the prediction is negative.
    :return: None
    """

    cf_found = False

    for curr_depth in range(1, max_depth + 1):

        counterfactuals_dict_list = []

        # Generate possible counterfactuals of given mutation depth
        for _ in range(nb_tries_per_depth):
            counterfactuals_dict_list.append(random_mutation(img_entry, curr_depth, shapes, colors, empty_probability))

        # Create the temporary directory and store the path
        temp_dir = tempfile.mkdtemp()

        # Saving the original image
        og_img_path = join(temp_dir, "original.png")
        gen_img(og_img_path, img_entry["content"], img_entry["division"], img_entry["size"])

        possible_cf_paths = []
        # Writing the possible counterfactuals into the temporary directory
        for cf_id, cf_dict in enumerate(counterfactuals_dict_list):
            curr_cf_path = join(temp_dir, str(cf_id) + ".png")
            possible_cf_paths.append(curr_cf_path)
            gen_img(curr_cf_path, cf_dict["content"], cf_dict["division"], cf_dict["size"])

        # Writing a CSV dataset of counterfactuals to be able to pass them through the model
        dataset_path = os.path.join(temp_dir, "dataset.csv")
        y_cf = [None for _ in range(len(possible_cf_paths))] # These values will not be used so they are set to None
        csv_content_train = np.array([np.concatenate((["path"], possible_cf_paths), axis=0),
                                      np.concatenate((["class"], y_cf), axis=0)]).T
        with open(dataset_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_content_train)

        # Verifying that the generated samples are actual counterfactuals according to the model
        dataset = get_dataset_transformed(db_dir, model_dir, dataset_path)
        _, _, y_pred_cf, _ = _predict(model_dir, device, dataset)
        actual_cf_idx_list = []
        for y_pred_cf_idx, y_pred_curr_cf in enumerate(y_pred_cf):
            # Exclusive or to check that the values are different
            if bool(y_pred) ^ bool(y_pred_curr_cf):
                actual_cf_idx_list.append(y_pred_cf_idx)
                cf_found = True
                break

        # Copying the actual original image into the XAI output directory
        shutil.copyfile(og_img_path, os.path.join(xai_output_path, str(sample_idx) + "_og.png"))

        # Generating the output for the first counterfactual found, providing at least one was found
        if cf_found:
            cf_dict = counterfactuals_dict_list[actual_cf_idx_list[0]]
            coords_diff = get_coords_diff(PatImgObj(img_entry), PatImgObj(cf_dict))
            cf_img = gen_img(None, cf_dict["content"], cf_dict["division"], cf_dict["size"],
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
            _generate_displayable_explanation(y_pred, cf_img, yes_pred_img_path, no_pred_img_path,
                                              output_img_AIonly_path,
                                              output_size=(600, 400), left_ratio=0.35, font_size=20, padding=5,
                                              AI_only=True)

            # Exporting the content of the JSON explanation
            with open(os.path.join(xai_output_path, f"cf_{sample_idx}.json"), 'w', encoding='utf-8') as f:
                json.dump(cf_dict, f, ensure_ascii=False, indent=4)

            # Returning the depth that was needed to obtain a counterfactual explanation
            return curr_depth

        # Removing the temporary dictionary
        shutil.rmtree(temp_dir)

    # In case no counterfactual was found at any depth, returning None
    return None

def generate_counterfactuals_resnet18_random_approach(db_dir, dataset_filename, model_dir, xai_output_path,
                                                      yes_pred_img_path, no_pred_img_path, shapes, colors,
                                                      empty_probability, max_depth, nb_tries_per_depth, devices,
                                                      n_jobs=-1, dataset_size=None, pos_pred_legend_path=None,
                                                      neg_pred_legend_path=None):
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
    :param dataset_filename: filename of the dataset.
    :param model_dir: path of the model directory.
    :param xai_output_path: path where to save the results.
    :param yes_pred_img_path: path the image that represents the yes prediction.
    :param no_pred_img_path: path the image that represents the no prediction.
    :param devices: List of devices to use for pytorch computation. Jobs are distributed to all devices.
    :param n_jobs: number of jobs to run in parallel.
    :param dataset_size: elements of the dataset are loaded until the size reaches this value. If None, the whole
    dataset is loaded.
    :param max_depth: maximum number of random mutations in the generated counterfactuals.
    :param nb_tries_per_depth: number of mutations which are assessed for each depth.
    :param shapes: list of possible shapes.
    :param colors: list of possible colors.
    :param empty_probability: probability of an empty cell.
    :param pos_pred_legend_path: path the legend when the prediction is positive.
    :param neg_pred_legend_path: path the legend when the prediction is negative.
    :return: None.
       """

    # Creating directories
    _create_dirs(xai_output_path)

    # Loading data
    dataset = get_dataset_transformed(db_dir, model_dir, dataset_filename, max_size=dataset_size)

    # Make prediction
    X, _, y_pred, model = _predict(model_dir, devices[0], dataset)

    # Load database
    db = load_db(db_dir)

    # Parallel computation of the images for the whole dataset.
    print("Generating counterfactual images")
    depths = Parallel(n_jobs=n_jobs)(delayed(_cf_single_sample_random_approach)(
        db_dir, i, xai_output_path, db[dataset.get_id(i)], y_pred[i], model_dir,
        devices[i%len(devices)], max_depth, nb_tries_per_depth, shapes, colors, empty_probability, yes_pred_img_path,
        no_pred_img_path, pos_pred_legend_path, neg_pred_legend_path) for i in tqdm.tqdm(range(len(X))))

    # Writing depths of generated counterfactuals
    csv_data = [
        {"CF_id": str(i), "depth": str(depths[i])} for i in range(len(X))
    ]
    with open(os.path.join(xai_output_path, "CF_generation.csv"), 'w', newline='') as csvfile:
        fieldnames = ['CF_id', 'depth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)


def create_xai_index(db_dir, dataset_filename, model_dir, xai_dirs, dataset_size, device):
    """
    Creating a csv file which contains the information of (dataset index, class, model prediction, path to image, [path to every xai generated image],
    path to the generated image for the prediction only).
    :param db_dir: root of the database.
    :param dataset_filename: filename of the dataset.
    :param model_dir: path of the model directory.
    :param xai_dirs: dictionary that contains for every explanation type, the path to where the explanations are saved.
    :param dataset_size: elements of the dataset are loaded until the size reaches this value. If None, the whole
    dataset is loaded.
    :param device: device to use for pytorch computation.
    :return:
    """

    # Loading data
    dataset = get_dataset_transformed(db_dir, model_dir, dataset_filename, max_size=dataset_size)

    # Make prediction
    X, y, y_pred, model = _predict(model_dir, device, dataset)

    csv_data = []

    for idx in range(len(dataset)):
        curr_line = {"idx": idx, "y": y[idx], "y_pred": y_pred[idx], "path": dataset.get_path(idx)}

        for xai_key, xai_dir in xai_dirs.items():
            curr_line[xai_key] = os.path.join(xai_dir, f"{idx}.png")
            curr_line["AI"] = os.path.join(xai_dir, f"{idx}_AIonly.png")

        csv_data.append(curr_line)

    with open(os.path.join(model_dir, "xai_index.csv"), 'w', newline='') as csvfile:
        fieldnames = ["idx", "y", "y_pred", "path", "AI"]
        fieldnames.extend([k for k in xai_dirs.keys()])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
