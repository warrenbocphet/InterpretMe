import glob

import cv2
import pytest

from read_bill_site.process.process import Extractor


test_data = []

img_paths = glob.glob(r"tests/img/*/*", recursive=True)
for path in img_paths:
    f = open(path + "/expected.txt")
    l = f.readline()
    expected_value = float(l)
    f.close()

    file_path = glob.glob(path + "/img*")[0]
    test_data.append((file_path, expected_value))


@pytest.mark.parametrize("input_path, input_expected", test_data)
def test_extractor_total_payable(input_path, input_expected):
    print(input_path)
    img = cv2.imread(input_path)

    ext = Extractor(img)
    ext.extract()

    assert ext.fields["total_payable"] == input_expected