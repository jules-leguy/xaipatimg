from PIL import Image, ImageDraw, ImageFont, __file__ as PIL_FILE
from PIL import Image, ImageDraw
from xaipatimg.datagen.dbimg import save_db
import math
import os
import tqdm
from joblib import Parallel, delayed

from . import COLORS_MAP
from . import SHAPES_MAP

def draw_circle(draw, x, y, size, color):
    draw.ellipse([x - size*0.8, y - size*0.8, x + size*0.8,
                 y + size*0.8], fill=color, outline=color)

def draw_square(draw, x, y, size, color):
    draw.rectangle([x - size*0.7, y - size*0.7, x + size*0.7,
                   y + size*0.7], fill=color, outline=color)

def draw_triangle(draw, x, y, size, color):
    half_size = size*0.8 / 2
    height = size*0.8 * math.sqrt(3) / 2
    points = [(x, y - height / 2),
              (x - half_size, y + height / 2),
              (x + half_size, y + height / 2)]
    draw.polygon(points, fill=color, outline=color)

def draw_cross(draw, x, y, size, color):
    # Thickness of the cross arms as a fraction of total size
    arm = size * 0.3
    half = size / 2
    half_arm = arm / 2

    # The cross is made of 12 points forming a “+” shape
    points = [
        (x - half_arm, y - half),     # top arm left
        (x + half_arm, y - half),     # top arm right
        (x + half_arm, y - half_arm), # upper right intersection
        (x + half,     y - half_arm), # right arm upper
        (x + half,     y + half_arm), # right arm lower
        (x + half_arm, y + half_arm), # lower right intersection
        (x + half_arm, y + half),     # bottom arm right
        (x - half_arm, y + half),     # bottom arm left
        (x - half_arm, y + half_arm), # lower left intersection
        (x - half,     y + half_arm), # left arm lower
        (x - half,     y - half_arm), # left arm upper
        (x - half_arm, y - half_arm)  # upper left intersection
    ]

    draw.polygon(points, fill=color, outline=color)

def draw_star(draw, x, y, size, color, points=4, inner_ratio=0.5):
    """
    Draws a symmetric star centered at (x, y).
    Rotating it by 90° (or 360/points * k) will produce the same appearance.
    """
    coords = []
    outer_radius = size*0.55
    inner_radius = size * inner_ratio

    # For perfect symmetry, start at 45° for 4-point stars
    angle_offset = -math.pi / 2 if points % 4 == 0 else -math.pi / 2

    for i in range(points * 2):
        angle = angle_offset + i * math.pi / points
        radius = outer_radius if i % 2 == 0 else inner_radius
        px = x + radius * math.cos(angle)
        py = y + radius * math.sin(angle)
        coords.append((px, py))

    draw.polygon(coords, fill=color, outline=color)

def draw_question_mark(draw, x, y, size, color, font_path=None):
    # Load a font
    if font_path:
        font = ImageFont.truetype(font_path, int(size))
    else:
        font = ImageFont.load_default(int(size))

    text = "?"

    # Get bounding box (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Center the text at (x, y)
    draw.text((x, y), "?", font=font, fill=color, anchor="mm")

def gen_img(img_path, content, division=(6, 6), dimension=(700, 700), to_highlight=None, draw_coordinates=True,
            empty_cell_as_question_mark=False, overwrite=False, return_image=False):
    """
    Generate a grid label image, as well as highlighting a feature in the image. 
    :param img_path: path where to save the generated image.
    :param content: data structure that describes the content of the image (coordinates and description of geometrical
    shapes in the picture.
    :param division: tuple that describes the number of horizontal and vertical divisions.
    :param dimension: tuple that describes the size of the image in pixels.
    :param to_highlight: List of (x, y) positions to highlight in the image.
    :param draw_coordinates: If True, draw coordinates on the image.
    :param empty_cell_as_question_mark: If True, draws a question mark in the center of the empty cells
    :param overwrite: whether to overwrite existing images. If False, no action will be taken if the image already exists.
    :param return_image : whether to return the generated image
    :return: None
    """
    to_highlight = set(to_highlight or [])

    if img_path is not None:
        # Exit if the file already exists and overwrite is set to False
        already_exists = os.path.exists(img_path)
        if already_exists and not overwrite:
            return None

        img_dir_path = os.path.dirname(img_path)
        os.makedirs(img_dir_path, exist_ok=True)

    # Create a blank white image
    img = Image.new("RGB", dimension, color="white")
    draw = ImageDraw.Draw(img)

    # Add padding for labels and define grid area
    font_size = 25
    padding = 50

    # define width and height insdide the padding, so subtract padding from both sides (left & right, top & bottom)
    grid_area_width = dimension[0] - padding * 2
    grid_area_height = dimension[1] - padding * 2
    grid_origin_x = padding
    grid_origin_y = padding

    # Define grid cell size
    cell_width = grid_area_width / division[0]
    cell_height = grid_area_height / division[1]

    # draw the grid line
    for i in range(division[0] + 1):          # verticals
        x0 = grid_origin_x + i * cell_width
        y0 = grid_origin_y
        x1 = x0
        y1 = grid_origin_y + grid_area_height
        draw.line([(x0, y0), (x1, y1)], fill="black", width=1)

    for j in range(division[1] + 1):          # horizontals
        x0 = grid_origin_x
        y0 = grid_origin_y + j * cell_height
        x1 = grid_origin_x + grid_area_width
        y1 = y0
        draw.line([(x0, y0), (x1, y1)], fill="black", width=1)

    if draw_coordinates:
        # draw labels
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except IOError:
            print("Warning: LiberationSans-Regular.ttf not found. Falling back to the default font.")
            font = ImageFont.load_default(size=font_size)

        # Columns: A, B, C, …
        for col in range(division[0]):
            label = chr(ord('A') + col)
            x = grid_origin_x + (col + 0.5) * cell_width   # column centre
            y = grid_origin_y - (padding / 2.2)
            draw.text((x, y), label, fill="gray", font=font, anchor="ms")

        # Rows: 1, 2, 3, …
        for row in range(division[1]):
            label = str(row + 1)
            x = grid_origin_x - (padding / 2.2)
            y = grid_origin_y + (row + 0.5) * cell_height   # row centre
            draw.text((x, y), label, fill="gray", font=font, anchor="rm")

    # Define shape size
    shape_size = 0.7 * min(cell_width, cell_height)

    non_empty_cells = []
    # Iterating over all the shapes to draw
    for c in content:

        # Extracting the features of the current shape to draw
        x, y = c["pos"]
        non_empty_cells.append((x, y))
        shape = SHAPES_MAP[c["shp"]]
        color = COLORS_MAP[c["col"]]

        # Calculate the center of the current grid cell
        x_center = grid_origin_x + (x + 0.5) * cell_width
        y_center = grid_origin_y + (y + 0.5) * cell_height

        # Draw the shape
        if shape == 'circle':
            draw_circle(draw, x_center, y_center, shape_size / 2, color)
        elif shape == 'square':
            draw_square(draw, x_center, y_center, shape_size / 2, color)
        elif shape == 'triangle':
            draw_triangle(draw, x_center, y_center, shape_size, color)
        elif shape == 'cross':
            draw_cross(draw, x_center, y_center, shape_size, color)
        elif shape == 'star':
            draw_star(draw, x_center, y_center, shape_size, color, points=8, inner_ratio=0.25)
    #integrating code by adding highlighting the symbols
    highlight_margin = 0.15 * shape_size
    for (x, y) in to_highlight:
        x_center = grid_origin_x + (x + 0.5) * cell_width
        y_center = grid_origin_y + (y + 0.5) * cell_height
        outer_r = shape_size / 2 + highlight_margin
        draw.ellipse(
            (x_center - outer_r, y_center - outer_r, x_center + outer_r, y_center + outer_r),
            outline=(153, 153, 153), width=3
        )

    # Drawing a question mark into every non-empty cell, if the option is on.
    if empty_cell_as_question_mark:
        for idx_x in range(division[0]):
            for idx_y in range(division[1]):
                if (idx_x, idx_y) not in non_empty_cells:
                    x_center = grid_origin_x + (idx_x + 0.5) * cell_width
                    y_center = grid_origin_y + (idx_y + 0.5) * cell_height

                    draw_question_mark(draw, x_center, y_center, shape_size / 2, "black")


    # Save the image
    if img_path is not None:
        img.save(img_path)

    if return_image:
        return img

    return None


def gen_img_and_save_db(db, db_dir, overwrite=False, draw_coordinates=True, empty_cell_as_question_mark=False,
                        do_save_db=True, n_jobs=1):
    """
    Generate every image from the DB.
    :param db_dir: path to the root of the DB
    :param db: database of image information and location
    :param overwrite: whether to overwrite existing images. If False, the images that already exist in the filesystem
    are ignored by this function.
    :param draw_coordinates: If True, draw coordinates on the image.
    :param empty_cell_as_question_mark: If True, draw a question mark in the center of the empty cells.
    :param do_save_db: whether to save the JSON file in addition to generating the images
    :param n_jobs: number of jobs to run in parallel.
    :return:
    """

    # Parallel generation of the images
    Parallel(n_jobs=n_jobs)(delayed(gen_img)(os.path.join(db_dir, v["path"]), v["cnt"], v["div"], v["size"],
                                             None, # to_highlight
                                             draw_coordinates, # draw_coordinates
                                             empty_cell_as_question_mark, # empty_cell_as_question_mark
                                             overwrite, # overwrite
                                             False # return_image
                                             ) for k, v in tqdm.tqdm(db.items()))
    if do_save_db:
        save_db(db=db, db_dir=db_dir)