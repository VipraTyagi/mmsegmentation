import os
import cv2
import matplotlib.pyplot as plt

def visualize_with_labels(vis_dir):
    # List all .png files in the given directory
    files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
    
    for idx, file in enumerate(files):
        image_path = os.path.join(vis_dir, file)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {file}")
            continue

        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img_rgb.shape
        
        half_width = width // 2
        gt_img = img_rgb[:, :half_width, :].copy()
        pred_img = img_rgb[:, half_width:, :].copy()

        # Overlay text
        cv2.putText(gt_img, 'Ground Truth', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_img, 'Prediction', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Create a figure
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_img)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img)
        plt.axis('off')
        
        plt.suptitle(file)

        # Save figure to disk instead of showing
        out_filename = os.path.join(vis_dir, f"labeled_{idx}.png")
        plt.savefig(out_filename, bbox_inches='tight')
        print(f"Saved {out_filename}")
        
        # Close the figure to avoid keeping them all in memory
        plt.close()

if __name__ == '__main__':
    vis_dir = '/home/vipra/Thesis/Semantic_Segmentation/results/sugarbeetchange/vis_data/vis_image'
    visualize_with_labels(vis_dir)
