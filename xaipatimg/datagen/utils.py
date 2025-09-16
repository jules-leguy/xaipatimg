import copy
import random

import numpy as np


def gen_rand_sym(shapes, colors):
    """
    Generating a random symbol with a shape drawn from the given array of shapes, and the color drawn from the given array
    of color.
    :param shapes: list of possible shapes
    :param colors: list of possible colors
    :return: random symbol image description
    """
    return {"shape": np.random.choice(shapes), "color": np.random.choice(colors)}


def get_coords_diff(patimgobj1, patimgobj2):
    """
    Returning the list of coordinates where the two given PatImgObj instances are not identical. Assuming both instances
    have the same grid size.
    :param patimgobj1: first instance of PatImgObj.
    :param patimgobj2: second instance of PatImgObj.
    :return:
    """
    assert patimgobj1.division == patimgobj2.division
    coords_list = []
    for x in range(patimgobj1.division[0]):
        for y in range(patimgobj1.division[1]):
            if patimgobj1.img_content_arr[x][y] != patimgobj2.img_content_arr[x][y]:
                coords_list.append((x, y))
    return coords_list

def random_mutation(img_dict, depth, shapes, colors, empty_probability):
    """
    Performing random mutations in the given image.
    :param img_dict: dictionary description of the image.
    :param depth: number of random mutations.
    :param shapes: list of possible shapes.
    :param colors: list of possible colors.
    :param empty_probability: probability of an empty cell.
    :return:
    """

    obj = PatImgObj(img_dict)

    for _ in range(depth):

        # Random coordinate generation
        pos_x = np.random.choice(np.arange(0, obj.division[0]))
        pos_y = np.random.choice(np.arange(0, obj.division[1]))

        # Mutation
        if random.random() < empty_probability:
            obj.remove_symbol(pos_x, pos_y)
        else:
            obj.set_symbol(pos_x, pos_y, gen_rand_sym(shapes, colors))

    return obj.get_img_dict()

class PatImgObj:
    """ Representing XAI pattern images as a Python object to facilitate the edition of the content """

    def __init__(self, img_dict):
        self.path = img_dict['path']
        self.size = img_dict['size']
        self.division = img_dict['division']
        self.img_content = copy.deepcopy(img_dict['content'])

        self.img_content_arr = np.full(self.division, None)
        for c in self.img_content:
            self.img_content_arr[c["pos"][0]][c["pos"][1]] = c

    def set_symbol(self, posx, posy, value):
        """
        Setting the symbol at the given coordinates to the given content. The pos entry in value is overridden to match
        the coordinates (posx, posy).

        :param posx: x coordinate
        :param posy: y coordinate
        :param value: image content value
        :return: None
        """
        value["pos"] = (posx, posy)
        self.img_content_arr[posx][posy] = value

    def remove_symbol(self, posx, posy):
        """
        Removing the symbol at the given coordinates.
        :param posx: x coordinate
        :param posy: y coordinate
        :return:
        """
        self.img_content_arr[posx][posy] = None

    def change_shapes_of_line(self, pos_y, shape):
        """
        Changing all shapes of the given line to the given shape. Colors are kept the same
        :param pos_y: y coordinate
        :return:
        """
        for i in range(self.division[0]):
            if self.img_content_arr[i, pos_y] is not None:
                self.img_content_arr[i, pos_y]["shape"] = shape

    def get_empty_lines(self):
        """
        Returning the indices of lines without any symbol
        :return:
        """
        empty_lines = []
        for j in range(self.division[1]):
            empty_line = True
            for i in range(self.division[0]):
                if self.img_content_arr[i, j] is not None:
                    empty_line = False
                    break
            if empty_line:
                empty_lines.append(j)

        return empty_lines

    def get_img_dict(self):
        """
        Output of the object in the form that is stored in the database file.
        :return:
        """
        img_content = []
        for i in range(self.division[0]):
            for j in range(self.division[1]):
                if self.img_content_arr[i][j] is not None:
                    img_content.append(self.img_content_arr[i][j])

        return {
            "size": self.size,
            "division": self.division,
            "content": img_content,
            "path": self.path,
        }

    def get_symbols_by(self, shape=None, color=None):
        """
        Returns all symbols in the image that match a given shape and/or color.
        If shape is None, no shape constraint is applied.
        If color is None, no color constraint is applied.
        If both are None, all symbols are returned.
        :param shape: shape of the symbol
        :param color: color of the symbol
        :return list of symbols matching the given constraints
        """
        found_symbols = []
        num_columns = self.division[0]
        num_rows = self.division[1]

        for x in range(num_columns):
            for y in range(num_rows):
                current_symbol = self.img_content_arr[x, y]
                if current_symbol is None:
                    continue

                shape_match = (
                    shape is None or current_symbol['shape'] == shape)
                color_match = (
                    color is None or current_symbol['color'] == color)

                if shape_match and color_match:
                    found_symbols.append(current_symbol)

        return found_symbols

    def get_empty_cells(self, row=None):
        """
        Returns a list of (x, y) positions in the grid where there is no symbol.
        If row is given, only returns empty cells in that row.
        :param row: the row number to look at. If not given, all rows are checked.
        :return: 
        """
        empty_cells = []
        num_columns = self.division[0]
        num_rows = self.division[1]

        for x in range(num_columns):
            for y in range(num_rows):
                if self.img_content_arr[x, y] is None:
                    if row is None or y == row:
                        empty_cells.append((x, y))
        return empty_cells
