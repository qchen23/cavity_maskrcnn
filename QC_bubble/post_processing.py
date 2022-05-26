import cv2
import sys
import numpy as np
from utils import *
import os
import random


def unapply_mask(img, original_image, mask):
  # loc = np.where(mask == True)
  # for r, c in zip(loc[0], loc[1]):
  #   img[r, c, :] = original_image[r, c, :] 
  for i, j in zip(mask[0], mask[1]):

    img[i, j, :] = original_image[i, j, :]

def post_process_mask(event, x, y, flags, param):

  (original_image, img, masks, mask_ids, removed_masks, colors, threshold) = param
  unmasked_sets = set()
  if event == cv2.EVENT_LBUTTONDOWN:
    for mask_id in mask_ids[y][x]:
      if mask_id in removed_masks:
        removed_masks.remove(mask_id)
      else:
        removed_masks.add(mask_id)

      for i, j in zip(masks[mask_id][0], masks[mask_id][1]):
        for mask_id in mask_ids[i][j]: unmasked_sets.add(mask_id)

    for mask_id in unmasked_sets:
      unapply_mask(img, original_image, masks[mask_id])

    for mask_id in unmasked_sets:
      if mask_id not in removed_masks:
        my_apply_mask(img, masks[mask_id], colors[mask_id], threshold, 0.5)


if len(sys.argv) != 3:
  print("usage: python QC_bubble/post_processing.py checkpoint_dir threshold")
  sys.exit()

# read from command line arguments
checkpoint_dir = sys.argv[1]
threshold = int(sys.argv[2])

# make sure the checkpoint exists
if not os.path.isdir(checkpoint_dir):
  print("checkpoint_dir {} doesn't exit".format(checkpoint_dir))
  sys.exit()

# read image and masks


files = os.listdir(checkpoint_dir)
oimg = ""
for f in files:
  if "original_image" in f:
    oimg = f
    break

img = cv2.imread(checkpoint_dir + "/" + oimg)

masks = np.load(checkpoint_dir + "/masks.npy", allow_pickle=True)
binary_masks = loc_to_masks((img.shape[0], img.shape[1]), masks)
# masks = loc_to_masks((img.shape[0], img.shape[1]), np.load(checkpoint_dir + "/masks.npy", allow_pickle=True))


rows, cols = img.shape[0], img.shape[1]
colors = my_random_colors(masks.shape[0])
original_image = np.copy(img)

# mask_ids[i][j] contains a set of ids at location i,j
mask_ids = np.array([set() for _ in range(rows * cols)]).reshape(rows, cols)     
for i, mask in enumerate(masks):
  my_apply_mask(img, mask, colors[i], threshold, 0.5)
  for r, c in zip(mask[0], mask[1]):
    mask_ids[r,c].add(i)


removed_masks = set()
cv2.namedWindow("bubble")
cv2.setMouseCallback('bubble', post_process_mask, (original_image, img, masks, mask_ids, removed_masks, colors, threshold))

while True:
  cv2.imshow("bubble", img)
  if cv2.waitKey(1) == ord('e'):
    break

# we save info here
output_stat(checkpoint_dir + "/cnn_human_stat.csv", original_image, binary_masks, removed_masks, threshold = threshold)
cv2.imwrite(checkpoint_dir + "/cnn_human_maks_image.png", img)

