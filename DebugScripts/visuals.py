#!/usr/bin/env python3
"""
Semantic Segmentation Visualizer with Raw→Contiguous ID Remapping and Pixel Counts
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Initial known mapping ---
RAW_TO_ID = {0: 0, 255: 1}
ID_TO_RAW = {0: 0, 1: 255}

CLASS_NAMES = {
    0: "Background",
    1: "Vegetation",  # or "Foreground"
    2: "weed150",  # placeholder; will be auto-added if encountered
}

PALETTE = {
    0: (0, 0, 0),
    1: (0, 114, 189),
    2: (217, 83, 25),
}

# Normalize PALETTE to flat RGB tuples
PALETTE = {k: tuple(int(c) for c in np.array(v).reshape(-1)[:3]) for k, v in PALETTE.items()}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --- Utilities ---
def _det_color_for_id(class_id: int):
    rng = np.random.RandomState(class_id * 9973 + 17)
    return tuple(int(x) for x in rng.randint(0, 256, size=3))


def _normalize_mask_dtype(im: Image.Image) -> np.ndarray:
    """Load mask as ndarray and normalize common encodings to 8-bit."""
    if im.mode == "P":
        return np.array(im, dtype=np.uint8)
    if im.mode.startswith("I;16") or im.mode == "I":
        a16 = np.array(im, dtype=np.uint16)
        vals = a16.reshape(-1)
        if vals.size and (np.mean((vals % 256) == 0) > 0.95):
            return (a16 // 256).astype(np.uint8)
        return (a16 >> 8).astype(np.uint8)
    return np.array(im, dtype=np.uint8)


def _remap_ids(raw_ids: np.ndarray):
    """Map raw mask values to contiguous ids and update mappings."""
    global RAW_TO_ID, ID_TO_RAW, CLASS_NAMES, PALETTE
    uniq = np.unique(raw_ids)
    for v in uniq.tolist():
        if v not in RAW_TO_ID:
            new_id = max(RAW_TO_ID.values()) + 1 if RAW_TO_ID else 0
            RAW_TO_ID[v] = new_id
            ID_TO_RAW[new_id] = v
            CLASS_NAMES.setdefault(new_id, f"Label_{v}")
            if new_id not in PALETTE:
                PALETTE[new_id] = tuple(int(c) for c in np.array(_det_color_for_id(new_id)).reshape(-1)[:3])
    lut = np.zeros(max(int(uniq.max()) + 1, 256), dtype=np.int32)
    for rv, cid in RAW_TO_ID.items():
        if rv < len(lut):
            lut[rv] = cid
    remapped = lut[raw_ids]
    return remapped


def _load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    im = Image.open(path)

    # Paletted PNG -> already indices, 2D
    if im.mode == "P":
        arr = np.array(im, dtype=np.uint8)
        uniq = set(np.unique(arr).tolist())
        if uniq <= {0, 255}:
            arr = (arr > 0).astype(np.uint8) * 255
        return arr  # HxW

    # 16-bit grayscale -> normalize to 8-bit when it is id*256
    if im.mode.startswith("I;16") or im.mode == "I":
        a16 = np.array(im, dtype=np.uint16)
        vals = a16.reshape(-1)
        if vals.size and (np.mean((vals % 256) == 0) > 0.95):
            arr = (a16 // 256).astype(np.uint8)
        else:
            arr = (a16 >> 8).astype(np.uint8)
        uniq = set(np.unique(arr).tolist())
        if uniq <= {0, 255}:
            arr = (arr > 0).astype(np.uint8) * 255
        return arr  # HxW

    # Plain 8-bit grayscale
    if im.mode == "L":
        arr = np.array(im, dtype=np.uint8)
        uniq = set(np.unique(arr).tolist())
        if uniq <= {0, 255}:
            arr = (arr > 0).astype(np.uint8) * 255
        return arr  # HxW

    # RGB or RGBA color-coded masks -> pack to a single int per pixel
    arr = np.array(im.convert("RGB"), dtype=np.uint8)  # HxWx3
    packed = (
        (arr[..., 0].astype(np.uint32) << 16) |
        (arr[..., 1].astype(np.uint32) << 8) |
        (arr[..., 2].astype(np.uint32))
    ).astype(np.uint32)
    return packed  # HxW (packed color key)



def _match_image_for_mask(mask_path: str, images_dir: str) -> str:
    base = os.path.basename(mask_path)
    name, _ = os.path.splitext(base)
    for ext in IMG_EXTS:
        cand = os.path.join(images_dir, name + ext)
        if os.path.exists(cand):
            return cand
    for alt in [name.replace("_mask", "").replace("_label", "").replace("_labels", "")]:
        for ext in IMG_EXTS:
            cand = os.path.join(images_dir, alt + ext)
            if os.path.exists(cand):
                return cand
    return ""


def _collect_files(folder: str):
    return [
        os.path.join(folder, fn)
        for fn in sorted(os.listdir(folder))
        if os.path.splitext(fn)[1].lower() in IMG_EXTS
    ]


# --- Visualization ---
def visualize(masks_dir, images_dir, alpha, save_dir, dpi):
    mask_files = _collect_files(masks_dir)
    if not mask_files:
        print(f"No mask files found in {masks_dir}")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    idx = 0

    def show_index(i):
        nonlocal idx
        idx = max(0, min(i, len(mask_files) - 1))
        mpath = mask_files[idx]
        ipath = _match_image_for_mask(mpath, images_dir)
        if not ipath:
            print(f"[{idx+1}/{len(mask_files)}] No matching image for {mpath}")
            return None

        base = _load_image(ipath)
        raw_mask = _load_mask(mpath)
        ids = _remap_ids(raw_mask)

        overlay = np.zeros((*ids.shape, 3), dtype=np.uint8)
        uniq_ids = np.unique(ids)
        for k in uniq_ids.tolist():
            col = PALETTE.get(k, _det_color_for_id(k))
            col = np.array(col).reshape(-1)[:3]  # flatten to (3,)
            col = tuple(int(c) for c in col)
            overlay[ids == k] = col

        blended = (alpha * overlay.astype(np.float32) +
                   (1.0 - alpha) * base.astype(np.float32)).clip(0, 255).astype(np.uint8)

        fig = plt.figure(figsize=(10, 6), dpi=dpi)
        ax = plt.gca()
        ax.imshow(blended)
        ax.axis("off")

        handles = []
        for k in uniq_ids.tolist():
            raw_val = ID_TO_RAW.get(k, "?")
            name = CLASS_NAMES.get(k, f"id_{k}")
            count = np.sum(ids == k)
            handles.append(Patch(facecolor=np.array(PALETTE[k]) / 255.0,
                                 edgecolor="none",
                                 label=f"{k} (raw {raw_val}): {name} [{count} px]"))
        ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
        ax.text(0.99, 0.02, "n: next  p: prev  s: save  q: quit",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        fig.suptitle(f"{os.path.basename(ipath)} + {os.path.basename(mpath)}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(0.001)
        return fig, ipath

    # Interactive mode
    res = show_index(idx)
    if res is None:
        print("Nothing to display.")
        return
    fig, ipath = res

    def on_key(event):
        nonlocal idx, fig, ipath
        if event.key == "n":
            plt.close(fig)
            res2 = show_index(idx + 1)
            if res2:
                fig, ipath = res2
                fig.canvas.mpl_connect("key_press_event", on_key)
                plt.show()
        elif event.key == "p":
            plt.close(fig)
            res2 = show_index(idx - 1)
            if res2:
                fig, ipath = res2
                fig.canvas.mpl_connect("key_press_event", on_key)
                plt.show()
        elif event.key == "s":
            out_dir = save_dir if save_dir else os.path.join(os.path.dirname(images_dir), "viz_out")
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.splitext(os.path.basename(ipath))[0] + "_viz.png"
            fig.savefig(os.path.join(out_dir, out_name))
            print(f"Saved {out_name} -> {out_dir}")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# --- Main ---
def main():
    ap = argparse.ArgumentParser(description="Visualize semantic segmentation masks with raw→contiguous ID remapping and pixel counts.")
    ap.add_argument("--masks-dir", type=str,
                    default="/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/semantics",
                    help="Folder with mask images.")
    ap.add_argument("--images-dir", type=str,
                    default="/home/vipra/Thesis/Semantic_Segmentation/data/ugvbonn/images",
                    help="Folder with corresponding RGB images.")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha in [0,1].")
    ap.add_argument("--save-dir", type=str, default="", help="If set, save visualizations here instead of interactive view.")
    ap.add_argument("--dpi", type=int, default=110, help="Figure DPI for display/save.")
    args = ap.parse_args()

    visualize(args.masks_dir, args.images_dir, args.alpha, args.save_dir, args.dpi)


if __name__ == "__main__":
    main()
