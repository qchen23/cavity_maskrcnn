import os
import xml.etree
from numpy import zeros, asarray
import random
import tensorflow as tf
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import numpy as np
#ignore many warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

# Create an empty instance of the mrcnn.utils.Dataset
class BubbleDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):

        self.add_class("dataset", 1, "bubble")
        # self.add_class("dataset",2,"precipitate")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        # shuffle the images in the dataset
        # 80% training  20 % validation
        filenames = sorted(os.listdir(images_dir))
        random.seed(100)
        random.shuffle(filenames)
        
        total_num_training = len(filenames)
    
        if is_train:
            
            filenames = filenames[:int(0.8*total_num_training)]
            
        else: #for validation
            filenames = filenames[int(0.8*total_num_training)+1:]
            

        for filename in filenames:
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.npy'
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
        print(len(self.image_info))

    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)
        root = tree.getroot()

        boxes = list()
        class_type = list()
        # get the position of bounding box
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            class_type_each = box.find('class').text
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            class_type.append(class_type_each)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
    
        return boxes, width, height, class_type

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']

        masks = np.load(path)
        class_ids = list()

        for _ in range(masks.shape[2]):
            class_ids.append(self.class_names.index('bubble'))

        return masks, asarray(class_ids, dtype='int32')

class BubbleConfig(mrcnn.config.Config):
    NAME = "bubble_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2 # if only bubble,  num will be 2 (including background)
    # IMAGE_CHANNEL_COUNT = 3

    DETECTION_MAX_INSTANCES = 400
    MAX_GT_INSTANCES = 200

    # Can be larger in GPU
    STEPS_PER_EPOCH = 272
    VALIDATION_STEPS = 68

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 512
    RPN_NMS_THRESHOLD = 0.9
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 666
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0


def train(model_path, epoch):
    # Prepare the dataset for use
    train_set = BubbleDataset()
    train_set.load_dataset(dataset_dir='bubble_dataset', is_train=True)
    # can train bubble first and then adding precipitates
    train_set.prepare()

    valid_dataset = BubbleDataset()
    valid_dataset.load_dataset(dataset_dir='bubble_dataset', is_train=False)
    valid_dataset.prepare()

    bubble_config = BubbleConfig()

    model = mrcnn.model.MaskRCNN(mode='training', 
                                model_dir='./', 
                                config=bubble_config)

    
    model.load_weights(filepath='mask_rcnn_coco.h5', 
                    by_name=True, 
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    
    #The excluded layers are those responsible for producing the class probabilities, bounding boxes, and masks.


    
    model.train(train_dataset=train_set,
                val_dataset=valid_dataset,
                learning_rate=bubble_config.LEARNING_RATE,
                epochs=epoch,
                layers='heads') 

    # Save the trained weights
    model.keras_model.save_weights(model_path)

train('bubble_mask_rcnn_6.h5', 5)
