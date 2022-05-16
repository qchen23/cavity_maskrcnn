import cv2
import sys
import numpy as np
from utils import *

def dfs(r, c, mask, visited, path):

  rows, cols = mask.shape[0], mask.shape[1]

  if not mask[r][c]: return
  if visited[r][c]: return

  path.append((r, c))
  visited[r][c] = True

  # 4 locations to move
  if r - 1 >= 0: dfs(r - 1, c , mask, visited, path)
  if r + 1 < rows: dfs(r + 1, c, mask, visited, path)
  if c - 1 >= 0: dfs(r, c - 1, mask, visited, path)
  if c + 1 < cols: dfs(r, c + 1, mask, visited, path)

def bubble_overlap(mask):
  visited = np.full((mask.shape), False)
  locs = np.where(mask)
  rv = []
  for r, c in zip(locs[0], locs[1]):
    path = []
    if not visited[r][c]:
      dfs(r, c, mask, visited, path)
      m = np.full((mask.shape), False)
      for cell in path:
        m[cell[0], cell[1]] = True
      rv.append(m)
  return rv



if len(sys.argv) != 2:
  print("python bubble_overlap.py mask_name")
  sys.exit()

masks = np.load(sys.argv[1])
total_masks = []
for i in range(masks.shape[2]):
  total_masks = total_masks + bubble_overlap(masks[:, :, i])

total_masks = np.array(total_masks)
total_masks = np.moveaxis(np.array(total_masks),0,-1)

print(total_masks.shape)

np.save("test.npy", total_masks)


