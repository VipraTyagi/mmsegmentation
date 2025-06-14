import os
import shutil
import torch
import sys
sys.path.append('/home/vipra/Thesis/Semantic_Segmentation/mmsegmentation')
from mmseg.apis import init_model
from mmseg.apis.inference import inference_model

# =============================================================================
# CONFIGURATION — adjust these paths to match your project
# =============================================================================
CONFIG_PATH   = '/home/vipra/Thesis/Semantic_Segmentation/mmsegmentation/configs/thesisdata/uavbonn.py'
CHECKPOINT    = '/home/vipra/Thesis/Semantic_Segmentation/experiments/phenobench/iter_32000.pth'
SRC_IMG_DIR   = '/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn/images/train'
SRC_MASK_DIR  = '/home/vipra/Thesis/Semantic_Segmentation/data/uavbonn/masks/train'
OUTPUT_DIR    = '/home/vipra/Thesis/Semantic_Segmentation/alData'
NUM_TO_SELECT = 50

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
model = init_model(CONFIG_PATH, CHECKPOINT, device='cuda:0')
# cfg = model.cfg
# cfg.test_dataloader.dataset.pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='PackSegInputs')
# ]
# model.test_cfg = cfg.test_dataloader.dataset
# =============================================================================
# FUNCTION: compute_image_entropy
# -----------------------------------------------------------------------------
# Runs a forward pass on a single image, obtains per-pixel class probabilities,
# computes pixel-wise entropy, then returns the mean entropy (uncertainty) score.
# =============================================================================
def compute_image_entropy(image_path: str) -> float:
    """
    Inference_model returns a SegDataSample. Its pred_sem_seg field holds a 
    DataContainer with shape [1, num_classes, H, W] containing raw logits.
    """
    result = inference_model(model, image_path)
    logits = result.pred_sem_seg.data[0]          # torch.Tensor [C,H,W]
   
    probs = torch.softmax(logits.float(), dim=0)           # convert to probabilities
    
    entropy_map = - (probs * probs.log()).sum(dim=0)
    return float(entropy_map.mean().item())

# =============================================================================
# MAIN SELECTION LOOP
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entropy_scores = []

    # Compute entropy score for every image in SRC_IMG_DIR
    for fname in sorted(os.listdir(SRC_IMG_DIR)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(SRC_IMG_DIR, fname)
        score = compute_image_entropy(img_path)
        entropy_scores.append((score, fname))
        print(f"Processed {fname} — Entropy: {score:.4f}")

    # Sort by descending entropy (highest uncertainty first)
    entropy_scores.sort(key=lambda x: x[0], reverse=True)

    # Select top-N uncertain samples
    selected = entropy_scores[:NUM_TO_SELECT]
    print(f"\nSelected {NUM_TO_SELECT} most uncertain images:")

    for score, fname in selected:
        print(f"  {fname}: {score:.4f}")
        # Copy image + mask to OUTPUT_DIR
        shutil.copy(os.path.join(SRC_IMG_DIR, fname), os.path.join(OUTPUT_DIR, fname))
        mask_path = os.path.join(SRC_MASK_DIR, fname)
        if os.path.isfile(mask_path):
            shutil.copy(mask_path, os.path.join(OUTPUT_DIR, fname))

    print(f"\nAll selected samples copied to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
