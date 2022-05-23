import numpy as np
import pandas as pd
import cv2
import colorsys
import random
import os



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
def bubble_overlap(mask):
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


def get_masks(masks):

  total_masks = []
  for i in range(masks.shape[2]):
    total_masks = total_masks + bubble_overlap(masks[:, :, i])

  total_masks = np.array(total_masks)
  total_masks = np.moveaxis(np.array(total_masks),0,-1)

  return total_masks

def my_apply_mask(image, mask, color, threshold, alpha=0.5):

  for i, j in zip(mask[0], mask[1]):
    if threshold > 0 and image[i, j, 0] <= threshold: continue
    for c in range(3):
      image[i, j, c] = image[i, j, c] * (1 - alpha) + alpha * color[c] * 255
  return image



def my_random_colors(N, bright=True):
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


def mask_array_to_position_set(mask_array):
  loc = np.where(mask_array == True)
  s = [(r, c) for r, c in zip(loc[0], loc[1])]

  return set(s)


def similarity(set_1, set_2, IoU):
  union = len(set_1.union(set_2))
  intersection = len(set_1.intersection(set_2))
  sim = intersection / union
  if sim < IoU: 
    return False #Considered as two different bubbles
  else:
    return True #considered as a same bubble

def overlapping(set_1,set_2,overlapping_th):
  intersection = len(set_1.intersection(set_2))
  if intersection/ len(set_1) > overlapping_th or intersection/ len(set_2) > overlapping_th:
    return True  # Do exist overlapping
  else:
    return False

def confusion_maxtrix_by_jaccard_similarity(true_masks, pred_masks, verbose = 0):
  pred_sets = []

  if len(pred_masks.shape) >= 3:
    for i in range(pred_masks.shape[2]):
      pred_sets.append(mask_array_to_position_set(pred_masks[:, :, i]))

  # confusion matirix. Once we have this, we can compute f1_score
  true_pos = 0
  false_pos = 0
  true_neg = 1  # Consider the background as one mask
  false_neg = 0
  threshold = 0.7 #IoU

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
    return -1,-1
  else:
    return long_p, short_p

def draw_rotated_rect(cnt, binary_mask):
  rect = cv2.minAreaRect(cnt)
  width = rect[1][0]
  height = rect[1][1]
  long_p = max(width, height)
  short_p = min(width, height)

  area_binary_mask = len(np.where(binary_mask == 255)[0])
  area_mask_fitting =  long_p * short_p

  if area_mask_fitting / area_binary_mask >= 1.25 or area_mask_fitting / area_binary_mask >=1.25:
    return -1,-1
  else:
    return long_p, short_p



def output_stat(filepath, image, masks, removed_masks = set(), cv2_draw_type = "ellipse", threshold=-1):
  # image is the containing intensity 3D array
  # masks = (image_width, image_height,instance) containing only True or False
  # removed_masks = set(), containing unwilling id from 0 to N-1
  # savediris the directory to save all the statis-info

  long_p_lst = []
  short_p_lst = []
 

  for i in range(masks.shape[2]):
    if i in removed_masks:
      continue
    binary_mask = masks[:,:,i:i+1]
    # Be carful binary_mask has already changed
    cnt = get_cnt(image,binary_mask,threshold)
 
    if cv2_draw_type =='ellipse':
      long_p, short_p = draw_ellipse(cnt, binary_mask)
    else: 
      long_p, short_p = draw_rotated_rect(cnt, binary_mask)

    if long_p > 0:
      long_p_lst.append(long_p)
      short_p_lst.append(short_p)
      
    
  dict_r= {'long d/pixel': long_p_lst, 'short d/pixel':short_p_lst}
  frame = pd.DataFrame(dict_r)
  frame.to_csv(filepath, index= False)


def masks_to_loc(masks):
  all_locs = []
  for i in range(masks.shape[2]):
    mask = masks[:, : ,i]
    loc = np.where(mask)
    all_locs.append(np.where(mask))

  return np.array(all_locs, dtype=object)

def loc_to_masks(shape, locs):
  masks = np.full((shape[0], shape[1], len(locs)), False)

  for i, loc in enumerate(locs):
    for r, c in zip(loc[0], loc[1]):
      masks[r, c, i] = True
  return masks
