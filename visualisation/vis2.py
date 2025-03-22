from PIL import Image
import matplotlib.pyplot as plt

# Path to your concatenated image
img_path = '/home/vipra/Thesis/Semantic_Segmentation/results/phenobench/vis_data/vis_image/test_05-15_00053_P0030943.png_683.png'

# Open the image and ensure it is in RGB mode
img = Image.open(img_path).convert("RGB")
width, height = img.size

# Calculate the midpoint and split the image
midpoint = width // 2
ground_truth = img.crop((0, 0, midpoint, height))
prediction = img.crop((midpoint, 0, width, height))

# Display the images using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ground_truth)
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(prediction)
plt.title('Prediction')
plt.axis('off')

plt.tight_layout()
plt.savefig('output_visualisation.png')
