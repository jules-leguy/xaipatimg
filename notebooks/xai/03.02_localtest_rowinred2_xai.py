from xaipatimg.ml.xai import generate_counterfactuals_resnet18
from xaipatimg.datagen.utils import PatImgObj, gen_rand_sym
import random
from xaipatimg.datagen.dbimg import load_db
import numpy as np
from xaipatimg.ml.xai import generate_shap_resnet18
import os

os.environ["DATA"] = os.path.expanduser("~/")
db_dir = os.environ["DATA"] + "PatImgXAI_data/db0.1.5_6x6/"
test_dataset_filename = "red_in_row_2_test.csv"
model_dir = os.environ["DATA"] + "models/db0.1.5_6x6/red_in_row_2/"

db = load_db(db_dir)

generate_shap_resnet18(db_dir, test_dataset_filename, model_dir, "cuda:0", n_jobs=1, dataset_size=100, masker="ndarray")

def extract_red_in_row2(img_content):
    """
      Generates a detailed explanation for the 'red_in_row_2' rule.
      If row 2 (index 1) contains a red symbol
        →  Rule 1: triangles ≥ blue symbols
      Otherwise
        →  Rule 2: green symbols ≥ squares
      """
    summary = {
        'red_in_row_2': False,
        'num_triangles': 0,
        'num_blue': 0,
        'num_green': 0,
        'num_squares': 0
    }
    for c in img_content:
        if c['pos'][1] == 1 and c['color'] == '#F86C62':
            summary['red_in_row_2'] = True

        if c['shape'] == 'triangle':
            summary['num_triangles'] += 1
        if c['color'] == '#7AB0CD':
            summary['num_blue'] += 1

        if c['color'] == '#87C09C':
            summary['num_green'] += 1
        if c['shape'] == 'square':
            summary['num_squares'] += 1

    return summary


def apply_red_in_row_2(img_content):
    summary = extract_red_in_row2(img_content)

    if summary['red_in_row_2']:
        return summary['num_triangles'] >= summary['num_blue']

    else:
        return summary['num_green'] >= summary['num_squares']


# counterfactual logic


COLOR_RED = "#F86C62"
COLOR_BLUE = "#7AB0CD"
COLOR_GREEN = "#87C09C"
COLORS = ["#F86C62", "#7AB0CD", "#F4D67B", "#87C09C"]
SHAPES = ["circle", "square", "triangle"]
SHAPES_NO_TRIANGLE = ["circle", "square"]
SHAPES_NO_SQUARE = ["circle", "triangle"]
COLORS_NO_BLUE = [c for c in COLORS if c != COLOR_BLUE]
COLORS_NO_GREEN = [c for c in COLORS if c != COLOR_GREEN]


def red_inrow2_counterfactuals(img_entry, is_pos, nb_cf):
    """
    Generate counter-factual images for the ‘red-in-row-2’ dataset.
    ───────────────────────────────────────────────────────────────
    If the original prediction is positive (is_pos=True) we make it
    negative by *breaking* the currently-active rule; if it is
    negative we make it positive by *satisfying* the active rule.
    """
    output = []

    for _ in range(nb_cf):
        patimgobj = PatImgObj(img_entry)
        summ = extract_red_in_row2(patimgobj.get_img_dict()['content'])

        # helpers
        def red_still_in_row2():
            return any(c['pos'][1] == 1 and c['color'] == COLOR_RED
                       for c in patimgobj.get_img_dict()['content'])

        def add_symbol(shape_choices, colour_choices, row=None):
            empties = patimgobj.get_empty_cells(row)
            if empties:
                x, y = random.choice(empties)
                patimgobj.set_symbol(x, y,
                                     gen_rand_sym(shapes=shape_choices,
                                                  colors=colour_choices))

        def recolour(symbols, new_colours):
            """Randomly pick one symbol and recolour it (shape unchanged)."""
            if symbols:
                sym = random.choice(symbols)
                x, y = sym['pos']
                new_sym = dict(sym)
                new_sym['color'] = random.choice(new_colours)
                patimgobj.set_symbol(x, y, new_sym)

        if is_pos:
            # Rule 1 traingle >= blue
            if summ['red_in_row_2']:
                while True:
                    cur = extract_red_in_row2(
                        patimgobj.get_img_dict()['content'])
                    if cur['num_triangles'] < cur['num_blue']:
                        break
                    # Prefer increasing blue; else decrease triangles
                    non_triangle = [c for c in patimgobj.get_symbols_by()
                                    if c['shape'] != 'triangle']
                    if non_triangle:
                        recolour(non_triangle, [COLOR_BLUE])
                    else:
                        add_symbol(SHAPES_NO_TRIANGLE, [COLOR_BLUE])

                # otherwise just insert a red symbol
                if not red_still_in_row2():
                    add_symbol(SHAPES, [COLOR_RED], row=1)

            # if no red in row 2 apply Rule 2 green >= squares
            else:
                while True:
                    cur = extract_red_in_row2(
                        patimgobj.get_img_dict()['content'])
                    if cur['num_green'] < cur['num_squares']:
                        break
                    greens = patimgobj.get_symbols_by(color=COLOR_GREEN)
                    if greens:
                        recolour(greens, COLORS_NO_GREEN)
                    else:
                        add_symbol(['square'], COLORS)

        # for negative cases
        else:
            if summ['red_in_row_2']:
                while True:
                    cur = extract_red_in_row2(
                        patimgobj.get_img_dict()['content'])
                    if cur['num_triangles'] >= cur['num_blue']:
                        break
                    add_symbol(['triangle'], COLORS_NO_BLUE)

                if not red_still_in_row2():
                    add_symbol(SHAPES, [COLOR_RED], row=1)

            else:
                while True:
                    cur = extract_red_in_row2(
                        patimgobj.get_img_dict()['content'])
                    if cur['num_green'] >= cur['num_squares']:
                        break
                    # Either turn a square green or add a green symbol
                    squares = patimgobj.get_symbols_by(shape='square')
                    if squares:
                        recolour(squares, [COLOR_GREEN])
                    else:
                        add_symbol(SHAPES, [COLOR_GREEN])

        output.append(patimgobj.get_img_dict())

    return output


generate_counterfactuals_resnet18(db_dir, test_dataset_filename, model_dir, red_inrow2_counterfactuals, 5, None, 1, None)
