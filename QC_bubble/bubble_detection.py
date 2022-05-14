import os
import tensorflow as tf
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

  
  # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
  


def detect(model_path = "bubble_mask_rcnn.h5", images_path = [], nm_pixels = [],fitting_type=[],saveframe_address_dir='outputs/'):

  model = mrcnn.model.MaskRCNN(mode="inference", 
                               config=SimpleConfig(),
                               model_dir=os.getcwd())

  model.load_weights(filepath=model_path,
                     by_name=True)

  range_lst = [512,1024,1536,2048,2560,3072]
  range_lst = np.arange(512, 3073, 64)

  frame_index = 1
  best_saveframe_address = ''
  best_info = None
  total_bubble_set ={}
 

  for img, nm_pixel, fitting_type  in zip(images_path, nm_pixels, fitting_type):
    print("Processing image ", img)
    num_bubbles = -1
    best_address = ""
    checkpoint_dir = saveframe_address_dir + "/checkpoint-{}/".format(frame_index)
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)

    for item in range_lst:

      checkpoint_address = checkpoint_dir + '/{}-{}'.format(frame_index,item)
      saveframe_address = saveframe_address_dir + '/{}-{}'.format(frame_index,item)
      info = ExtractMask(img, nm_pixel, fitting_type, item)
      # bubble amount and mask of each config
      detected_bubbles, per_mask = info.extract_masks(model, False)


      pre_size = len(total_bubble_set)
      for new_item in per_mask:
        new_item = mask_array_to_position_set(new_item)
        is_add = True
        
        for i in range(pre_size):
          old_item = total_bubble_set[i]
          # similarity ==False ==two bubbles, overlapping ==False
          if similarity(new_item, old_item,0.7)==True or overlapping(new_item,old_item,0.7)==True:
            is_add= False
            break
        if is_add:
          total_bubble_set.add(new_item)


      # save info into checkpoint
      info.save_image(checkpoint_address + ".png")
      info.save_dataframe(checkpoint_address + ".csv")
      info.save_confusion_matrix(checkpoint_address + "-cm.txt")

      if num_bubbles < detected_bubbles: #just save the good dataframe
        best_info = info
        num_bubbles = detected_bubbles
        best_address = saveframe_address
    
    # save the best info
    best_info.save_image(best_address + ".png")
    best_info.save_dataframe(best_address + ".csv")
    best_info.save_confusion_matrix(best_address + "-cm.txt")


    print(len(total_bubble_set))

    # Test save the final image


    # fig, _ = mrcnn.visualize.display_instances(image=image, 
    #                             masks=np.array(total_bubble_set), 
    #                             title = 'number of detected bubbles:{}'.format(len(masks)),
    #                             display = display)

    frame_index += 1


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
  if intersection/len(set_1) > overlapping_th or intersection/ len(set_2)> overlapping_th:
    return True  # Do exist overlapping
  else:
    return False



class ExtractMask:
  def __init__(self, images_path,nm_pixel,fitting_type,dim_value):
    self.images_path = images_path
    self.nm_pixel = nm_pixel
    self.long_d_p = []
    self.short_d_p = []
    self.img_mask = []
    self.size = np.array([])
    self.type = fitting_type
    self.max_dim = dim_value
    self.min_dim = dim_value
    self.df = None
    self.fig = None
    self.cm_info = None
    

  def confusion_maxtrix_by_jaccard_similarity(self, true_masks, pred_masks):
    pred_sets = []

    if len(pred_masks.shape) >= 3:
      for i in range(pred_masks.shape[2]):
        loc = np.where(pred_masks[:, :, i] == True)
        s = [(r, c) for r, c in zip(loc[0], loc[1])]
        pred_sets.append(set(s))


    # confusion matirix. Once we have this, we can compute f1_score
    true_pos = 0
    false_pos = 0
    true_neg = 1  # Consider the background as one mask
    false_neg = 0
    threshold = 0.6 #IoU

    for i in range(true_masks.shape[2]):
      loc = np.where(true_masks[:, :, i] == True)
      mask_set = set([(r, c) for r, c in zip(loc[0], loc[1])])

      max_sim = 0
      # best_mask = -1
      pred_set = set()

      for counter, s in enumerate(pred_sets):

        union = len(mask_set.union(s))
        intersection = len(mask_set.intersection(s))
        sim = intersection / union

        if (sim > max_sim):
          max_sim = sim
          pred_set = s

      if max_sim >= threshold:
        pred_sets.remove(pred_set)
        true_pos += 1  # in true and pred mask
      else:
        false_neg += 1 # in true mask but not in pred mask


    false_pos = len(pred_sets) # in pred mask but not in true mask

    if true_pos + false_pos == 0:
      precision = 0
    else:
      precision = true_pos / (true_pos + false_pos)
    
    if true_pos + false_neg == 0:
      recall = 0
    else:
      recall = true_pos / (true_pos + false_neg)

    if precision + recall == 0:
      f1_score = 0
    else:
      f1_score =  2 * (precision * recall) / (precision + recall)

   

    return (true_pos, true_neg, false_pos, false_neg, precision, recall, f1_score)

  def save_dataframe(self, filepath):
    self.df.to_csv(filepath, index= False)
  
  def save_image(self, filepath):
    self.fig.savefig(filepath,dpi='figure',format=None, metadata=None,bbox_inches=None, 
                    pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)


  def save_confusion_matrix(self, filepath):

    if self.cm_info == None: return
    (true_pos, true_neg, false_pos, false_neg, precision, recall, f1_score) = self.cm_info
    with open(filepath, "w") as f:
      f.write("TP = {}, TN = {}, FP = {}, FN = {}, precision = {}, recall = {}, f1_score = {}\n".format(true_pos, true_neg, false_pos, false_neg, precision, recall, f1_score))
  
  # input an image path
  # return a frame of all masks in this image 
  def extract_masks(self, model, display = False):

    annot = (self.images_path[:-4] + ".npy").replace("images", "annots")
    image = cv2.imread(self.images_path)
    grey = image[:, : , 0:1]
    false_pos = 0
    
    model.config.MEAN_PIXEL = np.array([np.mean(grey)*0.85])
    model.config.IMAGE_MIN_DIM = self.min_dim
    model.config.IMAGE_MAX_DIM = self.max_dim
  
    r = model.detect(images=[grey],verbose=0)[0]
    print(r["masks"].shape)

    rois, masks, class_ids, scores =[],[],[],[]

    # r['mask'] shape (image_width, image_height, mask_sequence)
    for i in range(r['masks'].shape[2]):
      if self.extract_single_mask(r['masks'][:,:,i:i+1],grey):
        rois.append(r["rois"][i])
        masks.append(r['masks'][:,:,i])
        class_ids.append(r['class_ids'][i])
        scores.append(r['scores'][i])

    
    dict_r= {'long d/pixel': self.long_d_p, 'short d/pixel':self.short_d_p,'long d/nm':np.array(self.long_d_p)*self.nm_pixel,'short d/nm': np.array(self.short_d_p)*self.nm_pixel}
    frame = pd.DataFrame(dict_r)

    masks = np.moveaxis(np.array(masks),0,-1)
    print(masks.shape, len(frame))

    if os.path.exists(annot):
      true_masks = np.load(annot)
      self.cm_info = self.confusion_maxtrix_by_jaccard_similarity(true_masks, r["masks"])

    CLASS_NAMES = ["BG", "bubble"]

    self.fig, _ = mrcnn.visualize.display_instances(image=image, 
                                    boxes=np.array(rois), 
                                    masks=masks, 
                                    class_ids=np.array(class_ids), 
                                    class_names=CLASS_NAMES, 
                                    scores=np.array(scores),
                                    title = 'number of detected bubbles:{}'.format(len(frame)),
                                    display = display)
            

    self.df = frame
    return len(frame), masks


  def draw_ellipse(self, cnt,single_mask):
    ellipse = cv2.fitEllipse(cnt)
    # the return value is the rotated rectangle in which the ellipse is inscribed
    # It recalls the cv2.minAreaRect(), which returns a Box2D structure
    # containing (x,y) of the top-left corner, (width, height), angle of rotation
    top_left_corner = ellipse[0] #(x,y)
    width = ellipse[1][0]
    height = ellipse[1][1]
    long_d = max(width, height)
    short_d = min(width, height)

    area_single_mask =len(np.where(single_mask == 255)[0])
    area_mask_fitting = np.pi * 0.25 * long_d * short_d


    if area_mask_fitting / area_single_mask >= 1.25 or area_mask_fitting / area_single_mask >= 1.25:
      return -1,-1,None
    else:
      single_mask_fitting = cv2.ellipse(single_mask*0.001, ellipse,(255,255,255),2) # draw the fitting+ mask
      return long_d, short_d, single_mask_fitting

  def draw_rotated_rect(self, cnt, single_mask):
    rect = cv2.minAreaRect(cnt)
    width = rect[1][0]
    height = rect[1][1]
    long_d = max(width, height)
    short_d = min(width, height)

    area_single_mask = len(np.where(single_mask == 255)[0])
    area_mask_fitting =  long_d * short_d

    if area_mask_fitting / area_single_mask >= 1.25 or area_mask_fitting / area_single_mask >=1.25:
      return -1,-1,None
    else:
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      single_mask_fitting = cv2.drawContours(single_mask*0.001,[box],0,(255,255,255),2) # draw the fitting+ mask
      return long_d, short_d, single_mask_fitting


  def extract_single_mask(self, binary_mask, image): 
    
    # print(len(np.where(binary_mask == True)[0]))
    # mask = image * binary_mask
    # mask_init2 = mask[np.where(mask>0)]
    # # nums,bins,patches = plt.hist(mask_init2,bins=10,edgecolor='k',density=False)  
    # # th = bins[5]
    # th = 60
    # mask[np.where(mask<=th)]=0
    # mask[np.where(mask>th)]=255
    # binary_mask[np.where(mask<=th)] = False

    mask = binary_mask * 255

    contours,hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = []
    for c in contours:
      if len(c) > len(cnt):
        cnt = c 

    if len(cnt) <= 10: return False

    # find the mask area
    area= cv2.contourArea(cnt)
    # find the contour perimeter
    perimeter = cv2.arcLength(cnt,True)
    # draw_rotated_rect(cnt,mask)
    if self.type =='ellipse':
      long_d, short_d, mask_with_fitting = self.draw_ellipse(cnt, mask)
    else: 
      long_d, short_d, mask_with_fitting = self.draw_rotated_rect(cnt, mask)

    if long_d > 0:
      self.long_d_p.append(long_d)
      self.short_d_p.append(short_d)
      self.img_mask.append(mask_with_fitting)
      return True
    return False
    



# filenames = sorted(os.listdir("../bubble_dataset/images"))
# random.seed(23423)
# random.shuffle(filenames)
# filenames = filenames[int(0.8 * len(filenames)) : ]

# val_sets = [ "../bubble_dataset/images/" + f for f in filenames]

filenames = os.listdir("../otest_set/")
val_sets = [ "../otest_set/" + f for f in filenames][:1]
detect("../augmented_v5.h5", val_sets, [0.19] * len(val_sets), ['ellipse'] * len(val_sets), 'test_outputs_qq/')


# model_address, [image_paths],[nm_pixel_lst]
# detect("512_chao_test1.h5", ["bubble_dataset/images/00017.png"], [0.19], ['ellipse'], 'outputs/results/')



#["bubble_dataset/images/00328.png" ,"bubble_dataset/images/00005.png"]
#,"bubble_dataset/images/00015.png","bubble_dataset/images/00329.png"