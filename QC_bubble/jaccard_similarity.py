from utils import *
import sys


if len(sys.argv) != 3:
  print("usage: python jaccard_similarity true_mask_file pred_mask_file")
  sys.exit()


true_masks = np.load(sys.argv[1])
pred_masks = np.load(sys.argv[2], allow_pickle= True)
pred_masks = loc_to_masks((true_masks.shape[0], true_masks.shape[1]), pred_masks)


cm = confusion_maxtrix_by_jaccard_similarity(true_masks, pred_masks, True)
print(cm)