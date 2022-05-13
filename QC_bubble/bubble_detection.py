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


class SimpleConfig(mrcnn.config.Config):
  NAME = "bubble_inference"

  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  NUM_CLASSES = 2

  DETECTION_MIN_CONFIDENCE = 0.5
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

  range_lst =[512,1024,1536,2048,2560,3072]
  frame_index = 1
  num = 0
  best_saveframe_address = ''

  for img, nm_pixel, fitting_type  in zip(images_path, nm_pixels, fitting_type):
     
    for item in range_lst:

      saveframe_address = saveframe_address_dir +'{}-{}'.format(frame_index,item)

      info = ExtractMask(img, nm_pixel,fitting_type,item,saveframe_address)

      frame_single_mask = info.extract_masks(model, False)

      if num< len(frame_single_mask): #just save the good dataframe
        frame_single_mask.to_csv(saveframe_address+'.csv', index= False)
        num = len(frame_single_mask)
        best_saveframe_address = saveframe_address

    for item in range_lst:
      saveframe_address = saveframe_address_dir +'{}-{}'.format(frame_index,item)
      if saveframe_address != best_saveframe_address:
        os.remove(saveframe_address+'.png')
        if os.path.exists(saveframe_address+'.csv'):
          os.remove(saveframe_address+'.csv')

    frame_index += 1
      



      
class ExtractMask:
  def __init__(self, images_path,nm_pixel,fitting_type,dim_value,saveframe_address):
    self.images_path = images_path
    self.nm_pixel = nm_pixel
    self.long_d_p = []
    self.short_d_p = []
    self.img_mask = []
    self.size = np.array([])
    self.type = fitting_type
    self.max_dim = dim_value
    self.min_dim = dim_value
    self.saveframe_address = saveframe_address
    
    
  
  # input an image path
  # return a frame of all masks in this image 
  def extract_masks(self, model, display = False): 
    image = cv2.imread(self.images_path)
    grey = image[:, : , 0:1]
    


    model.config.MEAN_PIXEL = np.array([np.mean(grey)*0.9])
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
    print(masks.shape)

    CLASS_NAMES = ["BG", "bubble"]

    mrcnn.visualize.display_instances(image=image, 
                                    boxes=np.array(rois), 
                                    masks=masks, 
                                    class_ids=np.array(class_ids), 
                                    class_names=CLASS_NAMES, 
                                    scores=np.array(scores),
                                    title = 'number of detected bubbles:{}'.format(len(frame)),
                                    saveframe_address=self.saveframe_address,
                                    display = display)
            


    return frame


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


    if np.abs(area_mask_fitting - area_single_mask) > 1000:
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

    area_single_mask =len(np.where(single_mask == 255)[0])
    area_mask_fitting = np.pi * 0.25 * long_d * short_d

    if np.abs(area_mask_fitting - area_single_mask) > 1000:
      return -1,-1,None
    else:
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      single_mask_fitting = cv2.drawContours(single_mask*0.001,[box],0,(255,255,255),2) # draw the fitting+ mask
      return long_d, short_d, single_mask_fitting




  def extract_single_mask(self, binary_mask, image): 
    
    # print(len(np.where(binary_mask == True)[0]))
    mask = image * binary_mask
    mask_init2 = mask[np.where(mask>0)]
    # nums,bins,patches = plt.hist(mask_init2,bins=10,edgecolor='k',density=False)  
    # th = bins[5]
    th = 60
    mask[np.where(mask<=th)]=0
    mask[np.where(mask>th)]=255
    binary_mask[np.where(mask<=th)] = False

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
    


# model_address, [image_paths],[nm_pixel_lst]
detect("augmented_v5.h5", ["bubble_dataset/images/00005.png"], [0.19],['ellipse'],'outputs/results/')



#["bubble_dataset/images/00328.png" ,"bubble_dataset/images/00005.png"]
#,"bubble_dataset/images/00015.png","bubble_dataset/images/00329.png"