from utils import *
import sys
import cv2


if len(sys.argv) != 5:
  print("usage: python jaccard_similarity img_file true_mask_file pred_mask_file threshold")
  sys.exit()



img = cv2.imread(sys.argv[1])

true_masks = np.load(sys.argv[2], allow_pickle = True)
pred_masks = np.load(sys.argv[3], allow_pickle = True)
true_masks = loc_to_masks((img.shape[0], img.shape[1]), true_masks)
pred_masks = loc_to_masks((img.shape[0], img.shape[1]), pred_masks)

th = float(sys.argv[4])

cm = confusion_maxtrix_by_jaccard_similarity(true_masks, pred_masks, th)
print(cm)



cm = confusion_maxtrix_bbx_by_jaccard_similarity(true_masks, pred_masks, th, 1)
print(cm)