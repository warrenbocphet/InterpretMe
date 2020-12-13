import cv2
import pytesseract
import matplotlib.pyplot as plt

from read_bill_site.process.process import Extractor


img = cv2.imread(r"read_bill_site/tests/img/gas/1/img.jpg")

ext = Extractor(img)
ext.extract()
print(ext.fields)

plt.imshow(ext.img, cmap="gray")
plt.scatter(ext.data["pos"][:, 0], ext.data["pos"][:, 1])
plt.show()