import os
import tensorflow as tf
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
from utils import *


class SimpleConfig(mrcnn.config.Config):
  NAME = "bubble_inference"

  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  NUM_CLASSES = 2
  

  DETECTION_MIN_CONFIDENCE = 0.6
  RPN_NMS_THRESHOLD = 0.7
  IMAGE_RESIZE_MODE = "square"
  IMAGE_CHANNEL_COUNT = 1
  MEAN_PIXEL = np.array([127.5])
  DETECTION_MAX_INSTANCES = 500
  IMAGE_MIN_DIM = 512
  IMAGE_MAX_DIM = 512

  

def save_fig(fig, filepath):
  fig.savefig(filepath,dpi='figure',format=None, metadata=None,bbox_inches=None, 
                    pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

def save_confusion_matrix(filepath, confusion_matrix):
  (true_pos, true_neg, false_pos, false_neg, precision, recall, f1_score) = confusion_matrix
  with open(filepath, "w") as f:
    f.write("TP = {}, TN = {}, FP = {}, FN = {}, precision = {}, recall = {}, f1_score = {}\n".format(true_pos, true_neg, false_pos, false_neg, precision, recall, f1_score))


def detect(model_path = "bubble_mask_rcnn.h5", images_path = [],fitting_type=[],saveframe_address_dir='outputs/'):
  CLASS_NAMES =['BG','bubble']

  model = mrcnn.model.MaskRCNN(mode="inference", 
                               config=SimpleConfig(),
                               model_dir=os.getcwd())

  model.load_weights(filepath=model_path,
                     by_name=True)

  range_lst = [512,1024,1536,2048,2560,3072]
  # range_lst = np.arange(512, 3073, 512)
  # range_lst = [512,1024]

  frame_index = 1 
  for img_name,  fitting_type  in zip(images_path,  fitting_type):
    
    print("Processing image ", img_name)
    image = cv2.imread(img_name)
    total_bubble_set =[]
    total_roi, total_class_ids, total_scores, total_mask= [],[],[],[]

    # create checkpoint directory 
    checkpoint_dir = saveframe_address_dir + "/checkpoint-{}/".format(frame_index)
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)


    img_size_lst = range_lst
    img_max_dim = max(image.shape[0], image.shape[1])
    new_size = math.ceil(img_max_dim/64.0)*64

    # To various image size, we provide different re-scaling detection sequence
    if img_max_dim >range_lst[-1]: 
      img_size_lst = [new_size, new_size + 512 * 2, new_size + 512 * 1, new_size - 512, new_size - 512 * 2]
    else:
      img_size_lst = [new_size] + img_size_lst

    for item in img_size_lst:
      info = ExtractMask(img_name, fitting_type, item)     
      rois, masks, class_ids, scores = info.extract_masks(model)

      pre_size = len(total_bubble_set)

      if len(rois) == 0: continue

      for i in range(masks.shape[2]):
        new_item = mask_array_to_position_set(masks[:,:,i])
        is_add = True
        
        for j in range(pre_size):
          old_item = total_bubble_set[j]
          # similarity ==False ==two bubbles, overlapping ==False
          if similarity(new_item, old_item,0.7) == True or overlapping(new_item,old_item,0.7) == True:
            is_add= False
            break
        if is_add:
          total_bubble_set.append(new_item)
          total_roi.append(rois[i])
          total_class_ids.append(class_ids[i])
          total_scores.append(scores[i])
          total_mask.append(masks[:,:,i])
    

    total_mask= np.moveaxis(np.array(total_mask),0,-1)
    total_roi = np.array(total_roi)
    total_class_ids = np.array(total_class_ids)
    total_scores = np.array(total_scores)

    
    fig, _ = mrcnn.visualize.display_instances(image=image, 
                                boxes=total_roi, 
                                masks=total_mask, 
                                class_ids=total_class_ids, 
                                class_names=CLASS_NAMES, 
                                scores=total_scores,
                                title = 'number of detected bubbles:{}'.format(len(total_roi)),
                                display = False)
                              

    # we save the final figure, mask, and stat
    cv2.imwrite(checkpoint_dir + "/original_image.png", image)

    annot = (img_name[:-4] + ".npy").replace("images", "annots")

    if os.path.exists(annot):
      true_masks = np.load(annot)
      cm = confusion_maxtrix_by_jaccard_similarity(true_masks, total_mask)
      save_confusion_matrix(checkpoint_dir + "/cm.txt", cm)

    if len(total_class_ids) != 0:
      save_fig(fig, checkpoint_dir  + "/mask_image.png")

      
      np.save(checkpoint_dir  + "/masks.npy", masks_to_loc(total_mask))
      
      output_stat(checkpoint_dir + "/cnn_stat.csv", image, total_mask)

    frame_index += 1
    



class ExtractMask:
  def __init__(self, images_path, cv2_draw_type, dim_value):
    self.images_path = images_path
    self.dim_value = dim_value
    self.cv2_draw_type = cv2_draw_type


  
  # input an image path
  # return a frame of all masks in this image 
  def extract_masks(self, model):
    image = cv2.imread(self.images_path)
    grey = image[:, : , 0:1]
    false_pos = 0
    
    model.config.MEAN_PIXEL = np.array([np.mean(grey)*0.85])
    model.config.IMAGE_MIN_DIM = self.dim_value
    model.config.IMAGE_MAX_DIM = self.dim_value
  
    r = model.detect(images=[grey],verbose=0)[0]
    rois, masks, class_ids, scores =[],[],[],[]

    # r['mask'] shape (image_width, image_height, mask_sequence)
    for i in range(r['masks'].shape[2]):
      if self.extract_single_mask(r['masks'][:,:,i:i+1],grey):
        rois.append(r["rois"][i])
        masks.append(r['masks'][:,:,i])
        class_ids.append(r['class_ids'][i])
        scores.append(r['scores'][i])

    masks = np.moveaxis(np.array(masks),0,-1)


    return rois, masks, class_ids, scores

  # threshold can be applied to subtract the Fresnel fringes
  def extract_single_mask(self, binary_mask, image, threshold = -1): 
    cnt = get_cnt(image, binary_mask, threshold)
    if len(cnt) <= 10: return False

    if self.cv2_draw_type =='ellipse':
      long_d, short_d = draw_ellipse(cnt, binary_mask)
    else: 
      long_d, short_d = draw_rotated_rect(cnt, binary_mask)

    if long_d > 0: return True
    return False
    



filenames = sorted(os.listdir("../bubble_dataset/images"))
random.seed(23423)
random.shuffle(filenames)
filenames = filenames[int(0.8 * len(filenames)) : ]
val_sets = [ "../bubble_dataset/images/" + f for f in filenames]

# data_dir = "./new_data/"
# filenames = os.listdir(data_dir)
# val_sets = [ data_dir + f for f in filenames]

# val_sets = ["../bubble_dataset/images/00328.png"]
detect("../augmented_v5.h5", val_sets , ['ellipse'] * len(val_sets), 'val-set3/')
