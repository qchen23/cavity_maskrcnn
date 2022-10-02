from labelbox import Client
import numpy as np
import os
import cv2
import urllib
import sys

# create the directory
def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    img_dir = "{}/images".format(dir_name)
    annot_dir = "{}/annots".format(dir_name)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if not os.path.exists(annot_dir):
        os.makedirs(annot_dir)


# download  masks of one sample from labelbox as a list of a list of positions of each mask
def generate_one_sample(label, label_id, dir_name):
    label_id = str(label_id)
    annot_name = "{}/annots/{}.npy".format(dir_name, label_id.zfill(5))
    img_name = "{}/images/{}.png".format(dir_name, label_id.zfill(5))

    image_np = label.data.value # shape = (pixel,pixel,RGB)

    
    # compress the RGB to one grey channel. The majority of images have a size < (1024, 1024).
    image_grey = image_np[: 1024, :1024, 0]

    if len(label.annotations) == 0: return # no features are marked

    masks = []
    all_masks = []
    for i, annotation in enumerate(label.annotations):

        # read image 
        url = annotation.value.mask.url
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        im = cv2.imdecode(arr, -1) # 'Load it as it is'
        im = im[:, :, 1]

        # get mask
        index = np.where(im == 255)
        if len(index[0]) == 0: continue
        
        # make sure it's within range        
        if max(index[0]) >= 1024 or max(index[1]) >= 1024: continue

        all_masks.append(index)


    # write the masks and image out
    all_masks = np.array(all_masks, dtype=object)
    cv2.imwrite(img_name, image_grey)
    np.save(annot_name, all_masks)


def create_data(api_key, project_id, data_dir, starting_id, num_images):
    # Use a valid api key to properly connect to the Labelbox Client
    client = Client(api_key=api_key)
    project = client.get_project(project_id)
   
    # Export the labels to json mode
    labels = project.label_generator() 
    labels = labels.as_list()
    print("Entire dataset size {}".format(len(labels)))
    make_dir(data_dir)
    for i in range(starting_id, min(num_images + starting_id, len(labels))):
        print("Generate image {}".format(i))
        generate_one_sample(labels[i], i, data_dir)




API_KEY = "your_api_key"
project_id = 'your_project_id'


if len(sys.argv) != 3:
  print("usage: python QC_bubble/image_data_export.py starting_image num_images")
  sys.exit()

starting_id = int(sys.argv[1])
num_images = int(sys.argv[2])

create_data(API_KEY, project_id, "bubble_dataset", starting_id, num_images)






