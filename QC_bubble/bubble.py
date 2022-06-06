import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model
import random
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
from utils import *

class BubbleDataset(mrcnn.utils.Dataset):

  def load_dataset(self, dataset_dir, is_train=True, split = 0.8):
    self.add_class("dataset", 1, "bubble")

    images_dir = dataset_dir + '/images/'
    annotations_dir = dataset_dir + '/annots/'

    # Get image files and shuffle the train image

    filenames = sorted(os.listdir(images_dir))
    random.seed(23423) 
    random.shuffle(filenames)
    if is_train:
      filenames = filenames[: int(split * len(filenames))]
    else:
      filenames = filenames[int(split * len(filenames)) : ]

    for filename in filenames:
      image_id = filename[:-4]
      img_path = images_dir + filename
      ann_path = annotations_dir + image_id + '.npy'
      self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # if is_train and dataset_dir != "./rotate_dataset":
    #   self.load_dataset("./rotate_dataset", True, 1)

    print(len(self.image_info))
    return len(self.image_info)

  def extract_boxes(self, filename):
    tree = xml.etree.ElementTree.parse(filename)

    root = tree.getroot()

    boxes = list()
    for box in root.findall('.//bndbox'):
      xmin = int(box.find('xmin').text)
      ymin = int(box.find('ymin').text)
      xmax = int(box.find('xmax').text)
      ymax = int(box.find('ymax').text)
      coors = [xmin, ymin, xmax, ymax]
      boxes.append(coors)

    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height

  def load_image(self, image_id):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(self.image_info[image_id]['path'])
    # print(image.shape)
    image = image.reshape((image.shape[0], image.shape[1], 1))
    # # If grayscale. Convert to RGB for consistency.
    # if image.ndim != 3:
    #     image = skimage.color.gray2rgb(image)
    # # If has an alpha channel, remove it for consistency
    # if image.shape[-1] == 4:
    #     image = image[..., :3]
    return image

  def load_mask(self, image_id):

    
    info = self.image_info[image_id]
    img = skimage.io.imread(info['path'])

    masks = np.load(info['annotation'], allow_pickle = True)
    masks = loc_to_masks((img.shape[0], img.shape[1]), masks)
    masks = get_masks(masks)

    class_ids = list()

    for _ in range(masks.shape[2]):
      class_ids.append(self.class_names.index('bubble'))

    # for i in range(len(boxes)):
    #   box = boxes[i]
    #   row_s, row_e = box[1], box[3]
    #   col_s, col_e = box[0], box[2]
    #   masks[row_s:row_e, col_s:col_e, i] = 1
    #   class_ids.append(self.class_names.index('bubble'))
    return masks, asarray(class_ids, dtype='int32')

class BubbleConfig(mrcnn.config.Config):
  NAME = "bubble_cfg"

  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  
  NUM_CLASSES = 2
  IMAGE_CHANNEL_COUNT = 1

  # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
  # TRAIN_ROIS_PER_IMAGE = 512
  RPN_NMS_THRESHOLD= 0.9

  RPN_TRAIN_ANCHORS_PER_IMAGE = 64
  TRAIN_ROIS_PER_IMAGE = 128
  
  IMAGE_RESIZE_MODE = "crop"
  IMAGE_MIN_DIM = 512
  IMAGE_MAX_DIM = 512
  IMAGE_MIN_SCALE = 2.0
  
  MEAN_PIXEL = np.array([127.5])
  # If enabled, resizes instance masks to a smaller size to reduce
  # memory load. Recommended when using high-resolution images.
  USE_MINI_MASK = True
  #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
  MINI_MASK_SHAPE = (56, 56)

  DETECTION_MAX_INSTANCES = 200
  MAX_GT_INSTANCES = 200
  # DETECTION_MIN_CONFIDENCE = 0.5
  STEPS_PER_EPOCH = 272
  VALIDATION_STEPS = 68



def train(model_path = "bubble_mask_rcnn.h5", dataset = "bubble_dataset", epoch = 5):

  train_set = BubbleDataset()
  tsize = train_set.load_dataset(dataset_dir=dataset, is_train=True)
  train_set.prepare()

  valid_dataset = BubbleDataset()
  vsize = valid_dataset.load_dataset(dataset_dir=dataset, is_train=False)
  valid_dataset.prepare()

  bubble_config = BubbleConfig()

  print(tsize, vsize)
  bubble_config.STEPS_PER_EPOCH = tsize
  bubble_config.VALIDATION_STEPS = vsize

  model = mrcnn.model.MaskRCNN(mode='training', 
                               model_dir='./', 
                               config=bubble_config)

  model.load_weights(filepath='mask_rcnn_coco.h5', 
                     by_name=True, 
                     exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

  # model.load_weights(filepath='./chao_test/chao_test1.h5', by_name=True)

  model.train(train_dataset=train_set, 
              val_dataset=valid_dataset, 
              learning_rate=bubble_config.LEARNING_RATE / 10, 
              epochs=epoch, 
              layers="all")
              # (conv1)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)

  model.keras_model.save_weights(model_path)

train("no_border3.h5", "bubble_dataset", 40)

