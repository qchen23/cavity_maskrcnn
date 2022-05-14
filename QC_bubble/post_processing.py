import cv2
import sys
import numpy as np
from mrcnn.visualize import apply_mask, random_colors
from utils import *
import os


def unapply_mask(img, original_image, mask):
  loc = np.where(mask == True)
  for r, c in zip(loc[0], loc[1]):
    img[r, c, :] = original_image[r, c, :]  

def post_process_mask(event, x, y, flags, param):

  (original_image, img, masks, mask_ids, removed_masks, colors) = param
  if event == cv2.EVENT_LBUTTONDOWN:
    for mask_id in mask_ids[x][y]:
      if mask_id in removed_masks:
        apply_mask(img, masks[:, :, mask_id], colors[mask_id], 0.4)
        removed_masks.remove(mask_id)
      else:
        removed_masks.add(mask_id)
        unapply_mask(img, original_image, masks[:, :, mask_id])


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
masks = np.load(checkpoint_dir + "/masks.npy")


rows, cols = masks.shape[0], masks.shape[1]
colors = random_colors(masks.shape[2])
original_image = np.copy(img)

# mask_ids[i][j] contains a set of ids at location i,j
mask_ids = np.array([set() for _ in range(rows * cols)]).reshape(rows, cols)     
for i in range(masks.shape[2]):  
  apply_mask(img, masks[:, :, i], colors[i], 0.5)
  loc = np.where(masks[:, :, i] == True)
  for r, c in zip(loc[0], loc[1]):
    mask_ids[c][r].add(i)


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

