import re

import pytesseract
import numpy as np
import cv2


class ImagePreprocessing:
    """
    The purpose of this class is to contains all the necessary image preprocessing
    before we extract data from it.

    As of current, only converting to grayscale is necessary.
    """

    def __init__(self, img):
        self.img = img

    def process(self):
        self._to_rgb()
        self._to_grayscale()

        return self.img

    def _to_grayscale(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def _enlarge(self):
        self.img = cv2.resize(self.img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

    def _sharpen(self):
        # Laplace filter
        filtering = np.array([[0, 0, -1, 0, 0],
                              [0, -1, -2, -1, 0],
                              [-1, -2, 16, -2, -1],
                              [0, -1, -2, -1, 0],
                              [0, 0, -1, 0, 0]])

        self.img = cv2.filter2D(self.img, -1, filtering)

    def _to_rgb(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)


class Extractor:
    def __init__(self, img):
        if type(img) is bytes:
            img = np.fromstring(img, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.preprocessing = ImagePreprocessing(img)

        self.img = img
        self.data = None

        # fields should be able to extend in the future
        # such as "bsb" and "pay date" etc.
        self.fields = {
            "total_payable": None
        }

        # the patterns of each fields can be more than one
        # this is used to detect the line contains related info
        self.patterns = {
            "total_payable": ["total", "direct", "amount", "debit"]
        }

    def _img_to_data(self):
        if self.img is not None:
            self.data = pytesseract.image_to_data(self.img, output_type="dict")

            # Calculate the center of each bounding box
            # use numpy array for faster computation
            x = np.array(self.data['left']).reshape((-1, 1))
            y = np.array(self.data['top']).reshape((-1, 1))
            w = np.array(self.data['width']).reshape((-1, 1))
            h = np.array(self.data['height']).reshape((-1, 1))

            self.data["pos"] = np.concatenate((x + w/2, y + h/2), axis=1)

            # Find all box that contains number.
            self.data["is_number"] = np.zeros((len(self.data["text"]),))
            self.data["number"] = []

            for i, s in enumerate(self.data["text"]):
                # self.data["number"].append(re.findall(r"[-+]?\d*\.\d+|\d+", s))  # Accept both float and int
                cur_number = [float(f) for f in re.findall(r"\d+\.\d+", s)]  # Only accept float, avoid number such as
                # 2019, 2020 etc.

                # There can be multiple number in a data cell. Better store it as a list of number.
                self.data["number"].append(cur_number)
                self.data["is_number"][i] = len(self.data["number"][i]) > 0

    def _find_best_candidate(self, nominated_fields):
        # Process "total_payable" field
        if len(nominated_fields["total_payable"]) == 0:
            max_number = -1
            for numbers in self.data["number"]:
                for number in numbers:
                    if number > max_number:
                        max_number = number

            if max_number == -1:
                best_total_payable = None
            else:
                best_total_payable = max_number

        else:
            best_total_payable = max(nominated_fields["total_payable"])

        return {
            "total_payable": best_total_payable
        }

    def _extract_relevant_fields(self):
        def backup_extract_total_payable(data):
            # To find the total payable, do the following:
            # 1. Find the cell that match at least one pattern (which is s)
            # 2. Find the closest 5 cells that contains a (floating) number

            # Calculate the distance between current cell (s) to other cell
            nominated_total_payable = []
            dist = np.linalg.norm((data["pos"] - data["pos"][i]), axis=1)

            # sort the cell based on dist
            ind = np.argsort(dist)
            is_number_sorted = data["is_number"][ind]  # sort is_number array based on dist
            ind_is_number = ind[is_number_sorted == 1]  # get the (sorted, is number) ind

            if len(ind_is_number) > 0:
                # get the closest 5 cells that contains number because it is not possible
                # to only consider the closest cell
                closest_number_inds = ind_is_number[:2]
                for ind in closest_number_inds:
                    nominated_total_payable.extend(data["number"][ind])

            return nominated_total_payable

        nominated_fields = {}  # Nominated fields can have multiple values for one fields
        # but self.fields can only have one value for each field.

        for i, s in enumerate(self.data["text"]):
            for key in self.fields.keys():
                if key not in nominated_fields:
                    nominated_fields[key] = []

                if any([pattern in s.lower() for pattern in self.patterns[key]]):
                    if key == "total_payable":
                        # Create the parallel mask
                        v = self.data["pos"] - self.data["pos"][i]
                        angles = np.arctan2(v[:, 1], v[:, 0])
                        angles = np.abs(angles)
                        angles[angles > (np.pi/2)] = np.pi - angles[angles > (np.pi/2)]

                        mask_row = angles < 20/180*np.pi
                        mask_column = angles > 70/180*np.pi

                        if (not (np.any(mask_row) or np.any(mask_column))) and (len(nominated_fields[key]) == 0):
                            backup_extract_total_payable(self.data)

                        # Create to the right, below mask
                        mask_right = self.data["pos"][i][0] < self.data["pos"][:, 0]
                        mask_below = self.data["pos"][i][1] < self.data["pos"][:, 1]

                        if np.any(mask_right):
                            mask_row_right = np.bitwise_and(np.bitwise_and(mask_right, mask_row), self.data["is_number"] == 1)
                        else:
                            mask_row_right = np.bitwise_and(mask_row, self.data["is_number"] == 1)

                        if np.any(mask_below):
                            mask_column_below = np.bitwise_and(np.bitwise_and(mask_below, mask_column), self.data["is_number"] == 1)
                        else:
                            mask_column_below = np.bitwise_and(mask_column, self.data["is_number"] == 1)

                        # Find inds that satisfy the condition
                        ind_row_right = np.where(mask_row_right)[0]
                        ind_column_below = np.where(mask_column_below)[0]

                        if(ind_row_right.shape[0] + ind_column_below.shape[0]) == 0:
                            if len(nominated_fields[key]) == 0:
                                nominated_fields[key] = backup_extract_total_payable(self.data)

                            continue

                        if ind_row_right.shape[0] > 0:
                            dist_x = self.data["pos"][ind_row_right, 0] - self.data["pos"][i, 0]

                        if ind_column_below.shape[0] > 0:
                            dist_y = self.data["pos"][ind_column_below, 1] - self.data["pos"][i, 1]

                        if ind_row_right.shape[0] > 0 and ind_column_below.shape[0] > 0:
                            if np.min(dist_x) < np.min(dist_y):
                                ind_closest = ind_row_right[np.argmin(dist_x)]
                            else:
                                ind_closest = ind_column_below[np.argmin(dist_y)]

                        elif ind_row_right.shape[0] > 0:
                            ind_closest = ind_row_right[np.argmin(dist_x)]

                        elif ind_column_below.shape[0] > 0:
                            ind_closest = ind_column_below[np.argmin(dist_y)]

                        else:
                            raise NameError("Unexpected logic at _extract_relevant_fields")

                        nominated_fields["total_payable"].extend(self.data["number"][ind_closest])

        self.fields = self._find_best_candidate(nominated_fields)

    def extract(self):
        self.img = self.preprocessing.process()
        self._img_to_data()
        if self.data is not None:
            self._extract_relevant_fields()

