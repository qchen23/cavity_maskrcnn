import cv2
import sys
import numpy as np
from utils import *

if len(sys.argv) != 3:
  print("python read_mask.py img_name mask_name")
  sys.exit()


img = cv2.imread(sys.argv[1])

print(np.mean(img))

masks = np.load(sys.argv[2], allow_pickle = True)
colors = my_random_colors(masks.shape[0])

for mask, color in zip(masks, colors):
  my_apply_mask(img, mask, color, -1, 0.5)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.imwrite("READ_MASK.png", img)

