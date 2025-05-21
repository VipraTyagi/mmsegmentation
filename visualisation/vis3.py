import cv2
import numpy as np

# === Configuration: update these file paths as needed ===
image   = "/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn/images/test/sugar_f1_171009_02_subImages_2_frame1_crop3.png"      # Original image
gt_mask  = "/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn/images/test/sugar_f1_171009_02_subImages_2_frame1_crop3.png"      # Ground truth mask (binary image)
pred_mask = "/home/vipra/Thesis/Semantic_Segmentation/results/imagetest.png"    # Predicted mask (binary image)

# Resize masks if needed
height, width = image.shape[:2]
if gt_mask.shape != (height, width):
    gt_mask = cv2.resize(gt_mask, (width, height), interpolation=cv2.INTER_NEAREST)
if pred_mask.shape != (height, width):
    pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)

print("Unique values in ground truth mask:", np.unique(gt_mask))
print("Unique values in predicted mask:", np.unique(pred_mask))

# Optionally apply threshold if the predicted mask is not binary
_, pred_mask_binary = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)

# Create color versions of the masks
gt_color = np.zeros_like(image)
pred_color = np.zeros_like(image)

# Color the masks (green for GT, red for predicted)
gt_color[gt_mask > 0] = [0, 255, 0]
pred_color[pred_mask_binary > 0] = [0, 0, 255]

# Visualize the individual components
cv2.imshow("Original Image", image)
cv2.imshow("Ground Truth Mask", gt_mask)
cv2.imshow("Predicted Mask (binary)", pred_mask_binary)
cv2.imshow("Predicted Mask Colored", pred_color)
cv2.waitKey(0)
cv2.destroyAllWindows()