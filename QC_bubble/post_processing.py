import cv2
import sys
import numpy as np
from utils import *
import os
import colorsys
import random


def random_colors(N, bright=True):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  brightness = 1.0 if bright else 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  random.shuffle(colors)
  return colors


def unapply_mask(img, original_image, mask):
  # loc = np.where(mask == True)
  # for r, c in zip(loc[0], loc[1]):
  #   img[r, c, :] = original_image[r, c, :] 
  for i, j in zip(mask[0], mask[1]):
    img[i, j, :] = original_image[i, j, :]

def apply_mask(image, mask, color, alpha=0.5):

  for i, j in zip(mask[0], mask[1]):
    for c in range(3):
      image[i, j, c] = image[i, j, c] * (1 - alpha) + alpha * color[c] * 255
  return image


def post_process_mask(event, x, y, flags, param):

  (original_image, img, masks, mask_ids, removed_masks, colors) = param
  if event == cv2.EVENT_LBUTTONDOWN:
    for mask_id in mask_ids[y][x]:
      if mask_id in removed_masks:
        apply_mask(img, masks[mask_id], colors[mask_id], 0.4)
        removed_masks.remove(mask_id)
      else:
        removed_masks.add(mask_id)
        unapply_mask(img, original_image, masks[mask_id])


if len(sys.argv) != 2:
  print("usage: python QC_bubble/post_processing.py checkpoint_dir")
  sys.exit()

# read from command line arguments
checkpoint_dir = sys.argv[1]

# make sure the checkpoint exists
if not os.path.isdir(checkpoint_dir):
  print("checkpoint_dir {} doesn't exit".format(checkpoint_dir))
  sys.exit()

# read image and masks
img = cv2.imread(checkpoint_dir + "/original_image.png")

masks = np.load(checkpoint_dir + "/masks.npy", allow_pickle=True)
# masks = loc_to_masks((img.shape[0], img.shape[1]), np.load(checkpoint_dir + "/masks.npy", allow_pickle=True))


print(masks.shape, img.shape)
rows, cols = img.shape[0], img.shape[1]
colors = random_colors(masks.shape[0])
original_image = np.copy(img)

# mask_ids[i][j] contains a set of ids at location i,j
mask_ids = np.array([set() for _ in range(rows * cols)]).reshape(rows, cols)     
for i, mask in enumerate(masks):
  apply_mask(img, mask, colors[i], 0.5)
  for r, c in zip(mask[0], mask[1]):
    mask_ids[r,c].add(i)


removed_masks = set()
cv2.namedWindow("bubble")
cv2.setMouseCallback('bubble', post_process_mask, (original_image, img, masks, mask_ids, removed_masks, colors))

while True:
  cv2.imshow("bubble", img)
  if cv2.waitKey(1) == ord('e'):
    break

# we save info here
output_stat(checkpoint_dir + "/cnn_human_stat.csv", original_image, masks, removed_masks)
cv2.imwrite(checkpoint_dir + "/cnn_human_maks_image.png", img)

