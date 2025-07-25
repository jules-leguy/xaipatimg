import copy

import numpy as np
from networkx.classes import is_empty


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
