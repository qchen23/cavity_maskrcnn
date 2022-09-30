import numpy as np
import pandas as pd
import cv2
import colorsys
import random
import os
from mrcnn.utils import *



# find (x, y) locations that belongs to a given mask
def dfs(r, c, mask, visited):

  rows, cols = mask.shape[0], mask.shape[1]
  stack = []
  path = []

  stack.append((r, c))
  while len(stack) > 0:
    (r, c) = stack.pop()
    path.append((r, c))
    visited[r][c] = True

    # 4 locations to move
    if r - 1 >= 0 and mask[r-1][c] and not visited[r-1][c]: stack.append((r-1,c))
    if r + 1 < rows and mask[r+1][c] and not visited[r+1][c]: stack.append((r+1,c))
    if c - 1 >= 0 and mask[r][c-1] and not visited[r][c-1]: stack.append((r,c-1))
    if c + 1 < cols and mask[r][c+1] and not visited[r][c+1]: stack.append((r,c+1))
  
  return path

# Group masks by component.
# It's possible that human annotated multiple bubbles as one single bubble. 
def split_multiple_masks(mask):
  visited = np.full((mask.shape), False)
  locs = np.where(mask)
  rv = []
  for r, c in zip(locs[0], locs[1]):
    if not visited[r][c]:
      path = dfs(r, c, mask, visited)
      m = np.full((mask.shape), False)
      for cell in path:
        m[cell[0], cell[1]] = True
      rv.append(m)
  return rv



# get masks
def get_masks(masks):

  total_masks = []
  for i in range(masks.shape[2]):
    total_masks = total_masks + split_multiple_masks(masks[:, :, i])

  total_masks = np.array(total_masks)
  total_masks = np.moveaxis(np.array(total_masks),0,-1)

  return total_masks


# Apply mask onto image
def my_apply_mask(image, mask, color, threshold, alpha=0.5):

  for i, j in zip(mask[0], mask[1]):
    if threshold > 0 and image[i, j, 0] <= threshold: continue
    for c in range(3):
      image[i, j, c] = image[i, j, c] * (1 - alpha) + alpha * color[c] * 255
  return image


# Get random color
def my_random_colors(N, bright=True):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """

  brightness = 1.0 if bright else 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  random.seed(0x1237)
  random.shuffle(colors)
  return colors


# conver bool mask to a set of position
def mask_array_to_position_set(mask_array):
  loc = np.where(mask_array == True)
  s = [(r, c) for r, c in zip(loc[0], loc[1])]

  return set(s)


# return true if iou of two sets < IoU
# otherwise return false
def similarity(set_1, set_2, IoU):
  union = len(set_1.union(set_2))
  intersection = len(set_1.intersection(set_2))
  sim = intersection / union
  if sim < IoU: 
    return False #Considered as two different bubbles
  else:
    return True #considered as a same bubble

# return true if overlapping part > overlapping_th
# otherwise return false
def overlapping(set_1,set_2,overlapping_th):
  intersection = len(set_1.intersection(set_2))
  if intersection/ len(set_1) > overlapping_th or intersection/ len(set_2) > overlapping_th:
    return True  # Do exist overlapping
  else:
    return False



# compute the confrusion matrix by jaccard similarity based on bounding box
def confusion_maxtrix_bbx_by_jaccard_similarity(true_masks, pred_masks, threshold = 0.6, verbose = 0):
  true_box = extract_bboxes(true_masks)
  pred_box = extract_bboxes(pred_masks)

  true_pos = 0
  false_pos = 0
  true_neg = 1  # Consider the background as one mask
  false_neg = 0

  used = [False] * pred_box.shape[0]

  for tb in true_box:

    max_sim = 0
    tb_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
    ui = -1
    for i, pb in enumerate(pred_box):
      if used[i]: continue

      y1 = max(tb[0], pb[0])
      y2 = min(tb[2], pb[2])
      x1 = max(tb[1], pb[1])
      x2 = min(tb[3], pb[3])
      intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
      pb_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
      union = pb_area + tb_area - intersection
      sim = intersection / union
      if (sim > max_sim): 
        max_sim = sim
        ui = i

    if verbose > 0: print("sim: {}".format(max_sim)) 

    if max_sim >= threshold:
      used[ui] = True
      true_pos += 1  # in true and pred mask
    else:
      false_neg += 1 # in true mask but not in pred mask

  false_pos = used.count(False)
  # in pred mask but not in true mask

  if true_pos + false_pos == 0: precision = 0
  else: precision = true_pos / (true_pos + false_pos)
  
  if true_pos + false_neg == 0: recall = 0
  else: recall = true_pos / (true_pos + false_neg)

  if precision + recall == 0: f1_score = 0
  else: f1_score =  2 * (precision * recall) / (precision + recall)

  return (true_pos, true_neg, false_pos, false_neg, precision, recall, f1_score)
  

# compute the confrusion matrix by jaccard similarity based on the masks
def confusion_maxtrix_by_jaccard_similarity(true_masks, pred_masks, threshold = 0.6, verbose = 0):
  pred_sets = []

  if len(pred_masks.shape) >= 3:
    for i in range(pred_masks.shape[2]):
      pred_sets.append(mask_array_to_position_set(pred_masks[:, :, i]))

  # confusion matirix. Once we have this, we can compute f1_score
  true_pos = 0
  false_pos = 0
  true_neg = 1  # Consider the background as one mask
  false_neg = 0

  for i in range(true_masks.shape[2]):
    mask_set = mask_array_to_position_set(true_masks[:, :, i])

    max_sim = 0
    pred_set = set()

    for counter, s in enumerate(pred_sets):

      union = len(mask_set.union(s))
      intersection = len(mask_set.intersection(s))
      sim = intersection / union

      if (sim > max_sim):
        max_sim = sim
        pred_set = s

    if verbose > 0: print("sim: {}".format(max_sim)) 

    if max_sim >= threshold:
      pred_sets.remove(pred_set)
      true_pos += 1  # in true and pred mask
    else:
      false_neg += 1 # in true mask but not in pred mask


  false_pos = len(pred_sets) # in pred mask but not in true mask

  if true_pos + false_pos == 0: precision = 0
  else: precision = true_pos / (true_pos + false_pos)
  
  if true_pos + false_neg == 0: recall = 0
  else: recall = true_pos / (true_pos + false_neg)

  if precision + recall == 0: f1_score = 0
  else: f1_score =  2 * (precision * recall) / (precision + recall)

  return (true_pos, true_neg, false_pos, false_neg, precision, recall, f1_score)


# Get contour using opencv lib
def get_cnt(image, binary_mask, threshold = -1):

  # turn any pixel whose value > threshold to white pixel
  # turn any pixel whose value <= threshold to black pixel
  if threshold > 0:
    mask = image[:, :, 0] * binary_mask
    mask_init2 = mask[np.where(mask>0)]
    mask[np.where(mask <= threshold)] = 0 
    mask[np.where(mask > threshold)] = 255
    binary_mask[np.where(mask<=threshold)] = False

  # get cnt
  mask = binary_mask * 255
  contours,hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnt = []
  for c in contours:
    if len(c) > len(cnt):
      cnt = c
  return cnt

# fit a ellipse onto a mask
def draw_ellipse(cnt,binary_mask):
  ellipse = cv2.fitEllipse(cnt)
  # the return value is the rotated rectangle in which the ellipse is inscribed
  # It recalls the cv2.minAreaRect(), which returns a Box2D structure
  # containing (x,y) of the top-left corner, (width, height), angle of rotation
  top_left_corner = ellipse[0] #(x,y)
  width = ellipse[1][0]
  height = ellipse[1][1]
  long_p = max(width, height)
  short_p = min(width, height)

  area_binary_mask =len(np.where(binary_mask)[0])
  area_mask_fitting = np.pi * 0.25 * long_p * short_p


  if area_mask_fitting / area_binary_mask >= 1.25 or area_mask_fitting / area_binary_mask >= 1.25:
    return ellipse, long_p, short_p, -1
  else:
    return ellipse, long_p, short_p, abs(1 - area_mask_fitting / area_binary_mask)

# fit a rect onto a mask
def draw_rotated_rect(cnt, binary_mask):
  rect = cv2.minAreaRect(cnt)
  width = rect[1][0]
  height = rect[1][1]
  long_p = max(width, height)
  short_p = min(width, height)

  area_binary_mask = len(np.where(binary_mask)[0])
  area_mask_fitting =  long_p * short_p


  if area_mask_fitting / area_binary_mask >= 1.25 or area_mask_fitting / area_binary_mask >=1.25:
    return rect, long_p, short_p, -1
  else:
    return rect, long_p, short_p, abs(1 - area_mask_fitting / area_binary_mask)

# fit a circle onto a mask
def draw_circle(cnt, binary_mask):
  (x,y),radius = cv2.minEnclosingCircle(cnt)
  center = (int(x),int(y))
  radius = int(radius)
  return center, radius


# output the statistical info
def output_stat(filepath, image, masks, removed_masks = set(), threshold=-1):
  # image is the containing intensity 3D array
  # masks = (image_width, image_height,instance) containing only True or False
  # removed_masks = set(), containing unwilling id from 0 to N-1
  # savediris the directory to save all the statis-info

  long_p_lst, short_p_lst = [], []
  long_p_lst_ellipse, short_p_lst_ellipse = [], []
  long_p_lst_rect, short_p_lst_rect = [], []
  p_lst_circle = []

  rect = np.copy(image)
  ellipse = np.copy(image)
  circle = np.copy(image)
  

  for i in range(masks.shape[2]):
    if i in removed_masks:
      continue
    binary_mask = masks[:,:,i:i+1]
    # Be carful binary_mask has already changed
    cnt = get_cnt(image,binary_mask,threshold)
 
    e, long_p_e, short_p_e, ratio_e = draw_ellipse(cnt, binary_mask)
    r, long_p_r, short_p_r, ratio_r = draw_rotated_rect(cnt, binary_mask)
    center, radius = draw_circle(cnt, binary_mask)

    if ratio_e > 0:

      # ellipse 
      ellipse = cv2.ellipse(ellipse,e,(0,255,0),2)
      long_p_lst_ellipse.append(long_p_e)
      short_p_lst_ellipse.append(short_p_e)

      # rect 
      box = cv2.boxPoints(r)
      box = np.int0(box)
      rect = cv2.drawContours(rect,[box],0,(0,0,255),2)

      long_p_lst_rect.append(long_p_r)
      short_p_lst_rect.append(short_p_r)

      # circle
      p_lst_circle.append(radius)
      circle = cv2.circle(circle,center,radius,(255,0,0),2)


      # average of ellipse and rect
      if ratio_r > 0:
        long_p_lst.append((long_p_e + long_p_r) / 2)
        short_p_lst.append((short_p_e + short_p_r) / 2)
      else:
        long_p_lst.append(long_p_e)
        short_p_lst.append(short_p_e)


      
  dict_r= {'long d/pixel': long_p_lst, 'short d/pixel':short_p_lst}
  frame = pd.DataFrame(dict_r)
  frame.to_csv(filepath + "/cnn_stat_average.csv", index= False)

  dict_r= {'long d/pixel': long_p_lst_ellipse, 'short d/pixel':short_p_lst_ellipse}
  frame = pd.DataFrame(dict_r)
  frame.to_csv(filepath + "/cnn_stat_ellipse.csv", index= False)

  dict_r= {'long d/pixel': long_p_lst_rect, 'short d/pixel':short_p_lst_rect}
  frame = pd.DataFrame(dict_r)
  frame.to_csv(filepath + "/cnn_stat_rect.csv", index= False)


  dict_r= {'long d/pixel': p_lst_circle, 'short d/pixel': p_lst_circle}
  frame = pd.DataFrame(dict_r)
  frame.to_csv(filepath + "/cnn_stat_circle.csv", index= False)


  cv2.imwrite(filepath + "/rect.png", rect)
  cv2.imwrite(filepath + "/ellipse.png", ellipse)
  cv2.imwrite(filepath + "/circle.png", circle)

  return len(long_p_lst)


# convert bool mask to a list of mask positions
def masks_to_loc(masks):
  all_locs = []
  for i in range(masks.shape[2]):
    mask = masks[:, : ,i]
    loc = np.where(mask)
    all_locs.append(np.where(mask))

  return np.array(all_locs, dtype=object)

# convert mask location to bool mask
def loc_to_masks(shape, locs):
  masks = np.full((shape[0], shape[1], len(locs)), False)

  for i, loc in enumerate(locs):
    for r, c in zip(loc[0], loc[1]):
      masks[r, c, i] = True
  return masks
