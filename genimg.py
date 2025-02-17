from PIL import Image, ImageDraw
from dbimg import save_db
import math
import os
import tqdm
from joblib import Parallel, delayed

# Function to draw a circle
def draw_circle(draw, x, y, size, color):
    draw.ellipse([x - size*0.8, y - size*0.8, x + size*0.8, y + size*0.8], fill=color, outline=color)

# Function to draw a square
def draw_square(draw, x, y, size, color):
    draw.rectangle([x - size*0.7, y - size*0.7, x + size*0.7, y + size*0.7], fill=color, outline=color)

# Function to draw a triangle
def draw_triangle(draw, x, y, size, color):
    half_size = size*0.8 / 2
    height = size*0.8 * math.sqrt(3) / 2
    points = [(x, y - height / 2),
              (x - half_size, y + height / 2),
              (x + half_size, y + height / 2)]
    draw.polygon(points, fill=color, outline=color)

def gen_img(img_path, content, division=(4,7), dimension=(400, 700), overwrite=False):
    """
    Generate an image that fits the given features.
    :param img_path: path where to save the generated image.
    :param content: data structure that describes the content of the image (coordinates and description of geometrical
    shapes in the picture.
    :param division: tuple that describes the number of horizontal and vertical divisions.
    :param dimension: tuple that describes the size of the image in pixels.
    :param overwrite: whether to overwrite existing images. If False, no action will be taken if the image already exists.
    :return: None
    """

    # Exit if the file already exists and overwrite is set to False
    already_exists = os.path.exists(img_path)
    if already_exists and not overwrite:
        return

    img_dir_path = os.path.dirname(img_path)
    if not os.path.exists(img_dir_path):
        os.makedirs(img_dir_path)

    # Create a blank white image
    img = Image.new("RGB", dimension, color="white")
    draw = ImageDraw.Draw(img)

    # Define grid cell size
    cell_width = dimension[0] / division[0]
    cell_height = dimension[1] / division[1]

    # Define shape size
    shape_size = 0.7 * max(cell_width, cell_height)

    # Iterating over all the shapes to draw
    for c in content:

        # Extracting the features of the current shape to draw
        x, y = c["pos"]
        shape = c["shape"]
        color = c["color"]

        # Calculate the center of the current grid cell
        x_center = (x + 0.5) * cell_width
        y_center = (y + 0.5) * cell_height

        # Draw the shape
        if shape == 'circle':
            draw_circle(draw, x_center, y_center, shape_size / 2, color)
        elif shape == 'square':
            draw_square(draw, x_center, y_center, shape_size / 2, color)
        elif shape == 'triangle':
            draw_triangle(draw, x_center, y_center, shape_size, color)

    # Save the image
    img.save(img_path)

def gen_img_and_save_db(db, db_root_path, overwrite=False, n_jobs=1):
    """
    Generate every image from the DB.
    :param db_root_path: path to the root of the DB
    :param db: database of image information and location
    :param overwrite: whether to overwrite existing images. If False, the images that already exist in the filesystem
    are ignored by this function.
    :param n_jobs: number of jobs to run in parallel.
    :return:
    """
    img_data_list = list(db.values())

    Parallel(n_jobs=n_jobs)(delayed(gen_img)(os.path.join(db_root_path, img_data_list[i]["path"]),
                                             img_data_list[i]["content"],
                                             img_data_list[i]["division"], img_data_list[i]["size"],
                                             overwrite) for i in tqdm.tqdm(range(len(img_data_list))))
    # Parallel generation of the images

    save_db(db=db, db_root_path=db_root_path)