import copy
import random

import numpy as np

from xaipatimg.datagen.genimg import gen_img


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
        pos_x = int(np.random.choice(np.arange(0, obj.division[0])))
        pos_y = int(np.random.choice(np.arange(0, obj.division[1])))

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

        # Matrix that contains a short textual representation of the color and shape at each position (two cells with
        # the same color and shape will have the same representation, as opposite to self.img_content_arr.
        self.img_content_arr_shape_color = np.full(self.division, "", dtype="U100")

        for c in self.img_content:
            self.img_content_arr_shape_color[c["pos"][0]][c["pos"][1]] = c["color"] + c["shape"]


    def set_symbol(self, posx, posy, value):
        """
        Setting the symbol at the given coordinates to the given content. The pos entry in value is overridden to match
        the coordinates (posx, posy).

        :param posx: x coordinate
        :param posy: y coordinate
        :param value: image content value
        :return: None
        """
        value["pos"] = [posx, posy]
        self.img_content_arr[posx][posy] = value
        self.img_content_arr_shape_color[posx][posy] = value["color"] + value["shape"]

    def remove_symbol(self, posx, posy):
        """
        Removing the symbol at the given coordinates.
        :param posx: x coordinate
        :param posy: y coordinate
        :return:
        """
        self.img_content_arr[posx][posy] = None
        self.img_content_arr_shape_color[posx][posy] = ""


    def find_submatrix_positions(self, submatrix_content, submatrix_shape, consider_rotations=False, consider_adjacent_switches=False):
        """
        Return the positions where the given content submatrix us found in the full image object.
        :param submatrix_content: content of the submatrix
        :param submatrix_shape: shape of the submatrix
        :param consider_rotations: whether to consider that a rotation of the pattern is also a match. All the 90°, 180
         and -90° rotations are searched for.
        of the pattern are s
        :param consider_adjacent_switches: if True,
        :return:
        """

        def rotate(img_content, x_division, y_division, direction=None):
            """
            Rotating the content of the given image. If direction == 1, rotating right. If direction == -1, rotating left.
            Direction == 2 corresponds to a 180° rotation (=2 consecutive rotations to the left or right).
            :param img_content:
            :param direction:
            :return:
            """
            new_img_content = copy.deepcopy(img_content)

            for c in new_img_content:
                if direction == 1:
                    c["pos"] = [y_division - 1 - c["pos"][1], c["pos"][0]]
                elif direction == -1:
                    c["pos"] = [c["pos"][1], x_division - 1 - c["pos"][0]]
                elif direction == 2:
                    c["pos"] = [x_division - 1 - c["pos"][0], y_division - 1 - c["pos"][1]]

            return new_img_content

        # If consider_rotations is True, returning the concatenation of indices of the left rotation of the submatrix, the original submatrix, the
        # right rotation of the submatrix and the 180° rotation of the submatrix.
        if consider_rotations:
            return (self.find_submatrix_positions(rotate(submatrix_content, submatrix_shape[0], submatrix_shape[1], -1), submatrix_shape)
                    + self.find_submatrix_positions(submatrix_content, submatrix_shape)
                    + self.find_submatrix_positions(rotate(submatrix_content, submatrix_shape[0], submatrix_shape[1], 1), submatrix_shape)
                    + self.find_submatrix_positions(rotate(submatrix_content, submatrix_shape[0], submatrix_shape[1], 2), submatrix_shape))

        # Constructing the submatrix the same way the full matrix is constructed
        submatrix_np = np.full(submatrix_shape, "", dtype="U100")
        for c in submatrix_content:
            submatrix_np[c["pos"][0]][c["pos"][1]] = c["color"] + c["shape"]

        M, N = self.img_content_arr_shape_color.shape
        m, n = submatrix_np.shape
        positions = []

        for main_idx in range(M - m + 1):
            for main_idy in range(N - n + 1):

                og_window_matrix = np.array(self.img_content_arr_shape_color[main_idx:main_idx + m, main_idy:main_idy + n])
                window_matrices = [og_window_matrix]

                if consider_adjacent_switches:
                    pass
                #     for cell_x in range(m):
                #         for cell_y in range(n):
                #
                #             def is_exchange_valid(idx1, idy1, idx2, idy2):
                #                 """
                #                 Making sure that both cells are defined, both cells are on the same line or the same
                #                 column, and both cells contain a symbol
                #                 :return:
                #                 """
                #                 return ((idx1 >=0 and idx2 >=0 and idx1 < M and idx2 < M and
                #                     idy1 >=0 and idy2 >=0 and idy1 < N and idy2 < N)
                #                         and (idx1 == idx2 or idy1 == idy2)
                #                         and self.img_content_arr_shape_color[idx1][idy1]
                #                         and self.img_content_arr_shape_color[idx2][idy2])
                #
                #             for offset_x in [-1, 1]:
                #                 for offset_y in [-1, 1]:


                else:
                    window_matrices = [og_window_matrix]

                for window_matrix in window_matrices:

                    # Replacing every symbol from the window matrix that does not match with the expected symbols with
                    # an empty string. This allows comparing only the explicitly defined symbols of the submatrix. Thus,
                    # non defined symbols of the submatrix can be matched with any symbols of the window matrix.
                    for sub_idx in range(m):
                        for sub_idy in range(n):
                            if window_matrix[sub_idx][sub_idy] != submatrix_np[sub_idx][sub_idy]:
                                window_matrix[sub_idx][sub_idy] = ""

                    if np.array_equal(window_matrix, submatrix_np):
                        positions.append((main_idx, main_idy))

        return positions


    def change_shapes_of_line(self, pos_y, shape):
        """
        Changing all shapes of the given line to the given shape. Colors are kept the same
        :param pos_y: y coordinate
        :return:
        """
        for i in range(self.division[0]):
            if self.img_content_arr[i, pos_y] is not None:
                self.img_content_arr[i, pos_y]["shape"] = shape
                self.img_content_arr_shape_color[i][pos_y] = self.img_content_arr[i, pos_y]["color"] + shape


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
