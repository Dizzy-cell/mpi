# mpi_from_stereo.py
# pip install opencv-python numpy

import cv2
import numpy as np
import os

from IPython import embed

def compute_disparity(left, right, max_disp=128):
    """Compute disparity from left/right using OpenCV StereoSGBM (returns float32 disparity in px)."""
    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # StereoSGBM params (tweak for your imgs)
    min_disp = 0
    num_disp = max_disp - min_disp
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16
    window_size = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=min_disp,
                                 numDisparities=num_disp,
                                 blockSize=window_size,
                                 P1=8*3*window_size**2,
                                 P2=32*3*window_size**2,
                                 disp12MaxDiff=1,
                                 preFilterCap=63,
                                 uniquenessRatio=10,
                                 speckleWindowSize=100,
                                 speckleRange=32)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0  # OpenCV scales by 16
    # Replace negative / invalid disparities with 0
    disp[disp < 0] = 0.0
    return disp

def normalize_disp(disp):
    """Normalize disparity to [0,1]."""
    dmin = np.min(disp)
    dmax = np.max(disp)
    if dmax - dmin < 1e-6:
        return np.zeros_like(disp)
    return (disp - dmin) / (dmax - dmin)

def build_mpi_layers(left_img, disparity, num_planes=8, soft_sigma=0.05):
    """
    Build N MPI layers from left_img and normalized disparity (0..1).
    Returns list of (RGBA arrays) from far to near.
    soft_sigma: controls softness of assignment (relative to normalized disparity range)
    """
    H, W = disparity.shape
    # plane centers in normalized [0..1], far -> near (small disp -> far)
    # choose linear spacing
    plane_centers = np.linspace(0.0, 1.0, num_planes)
    layers = []
    # for each plane compute soft alpha weight via gaussian of (disp - center)
    for c in plane_centers:
        diff = disparity - c
        # gaussian weight as assignment strength
        alpha = np.exp(-0.5 * (diff / soft_sigma) ** 2)
        # normalize alpha across planes to avoid large overcover (optional)
        # We'll keep raw weights and then compute per-pixel normalization
        layers.append({'center': c, 'alpha_raw': alpha})
    # stack raw alphas shape (N,H,W)
    alpha_stack = np.stack([l['alpha_raw'] for l in layers], axis=0)
    # normalize per pixel so sum to <=1 (soft assignment)
    alpha_sum = np.sum(alpha_stack, axis=0) + 1e-8
    alpha_norm = alpha_stack / alpha_sum  # shape N,H,W
    rgba_layers = []
    # color from left image
    left_f = left_img.astype(np.float32) / 255.0
    for i in range(len(layers)):
        a = alpha_norm[i]  # HxW in (0,1)
        # create color image (H x W x 3) same across plane
        color = left_f.copy()
        # alpha channel: per-pixel soft alpha
        alpha8 = (a * 255.0).astype(np.uint8)
        # build RGBA uint8
        rgba = np.dstack(( (color * 255).astype(np.uint8), alpha8 ))
        rgba_layers.append(rgba)
    return rgba_layers, plane_centers

def inpaint_layer_alpha(rgba_layer):
    """Fill transparent holes in a layer using OpenCV inpaint on color (guided by alpha mask)."""
    # rgba_layer: H x W x 4 (uint8)
    color = rgba_layer[:, :, :3]
    alpha = rgba_layer[:, :, 3]
    mask = (alpha == 0).astype(np.uint8) * 255
    if mask.sum() == 0:
        return rgba_layer
    # inpaint expects 8-bit 1 channel mask and 3-channel color
    inpainted = cv2.inpaint(color, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # set alpha to original alpha (still zeros for previously empty)
    out = np.dstack((inpainted, alpha))
    return out

def save_layers(layers, out_dir="mpi_layers"):
    os.makedirs(out_dir, exist_ok=True)
    for i, rgba in enumerate(layers):
        fname = os.path.join(out_dir, f"layer_{i:02d}.png")
        # cv2.imwrite supports 4-channel PNG
        cv2.imwrite(fname, rgba)
    print(f"Saved {len(layers)} layers to {out_dir}")

def composite_layers(layers):
    """Composite RGBA layers (list of HxWx4 uint8) from far->near"""
    H, W, _ = layers[0].shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    out_alpha = np.zeros((H, W), dtype=np.float32)
    for rgba in layers:
        col = rgba[:, :, :3].astype(np.float32) / 255.0
        a = rgba[:, :, 3].astype(np.float32) / 255.0
        # alpha blending: out = col * a + out * (1-a)
        out = col * a[..., None] + out * (1.0 - a[..., None])
        out_alpha = a + out_alpha * (1.0 - a)
    out_img = np.clip((out * 255.0), 0, 255).astype(np.uint8)
    return out_img

def render_view_from_mpi(layers, plane_centers, disparity_norm, t=0.5, max_shift_px=20):
    """
    Render intermediate view param t in [0,1] where 0=left view, 1=right view.
    We shift each layer by plane disparity = center * max_shift_px scaled by t.
    Note: this is a simple parallax shift (horizontal). For better correctness,
    you should project using proper camera intrinsics.
    """
    H, W, _ = layers[0].shape
    canvas = np.zeros((H, W, 4), dtype=np.float32)  # RGBA float
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    for i, rgba in enumerate(layers):
        center = plane_centers[i]
        # compute shift in px for this plane (positive shift to right for t>0)
        shift = (center - 0.5) * 2.0 * max_shift_px * (t - 0.0)  # simple mapping: center=0.5->0 shift
        # create remap coordinates
        map_x = (grid_x + shift).astype(np.float32)
        map_y = grid_y.astype(np.float32)
        # remap color and alpha separately
        color = cv2.remap(rgba[:, :, :3], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) / 255.0
        alpha = cv2.remap(rgba[:, :, 3].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) / 255.0
        # composite into canvas (over)
        canvas[:, :, :3] = color * alpha[..., None] + canvas[:, :, :3] * (1.0 - alpha[..., None])
        canvas[:, :, 3] = alpha + canvas[:, :, 3] * (1.0 - alpha)
    out = np.clip(canvas[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    return out

# ---------------------------
# MAIN demo flow
# ---------------------------
def main():
    # load left/right
    left = cv2.imread("frame_2.png")  # provide left.png
    right = cv2.imread("frame_3.png")  # provide right.png
    if left is None or right is None:
        print("Place left.png and right.png in working dir.")
        return

    # 1) compute disparity (px)
    print("Computing disparity...")
    disp = compute_disparity(left, right, max_disp=128)  # tweak max_disp
    
    # cv2.imwrite("disparity.png", (disp / np.max(disp) * 255).astype(np.uint8))

    # disp2 = cv2.imread("disparity_depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # cv2.imwrite("disparity2.png", (disp2 / np.max(disp2) * 255).astype(np.uint8))
    #embed()

    # rgb + depth 

    left = cv2.imread("frame_1.png")
    disp = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # disp = cv2.GaussianBlur(disp, (5,5), 1.5)

    # kernel = np.ones((5,5), np.uint8)
    # dilated = cv2.dilate(disp, kernel, iterations=1)

    # 2) normalize disparity to 0..1
    disp_norm = normalize_disp(disp)

    # 3) build MPI layers
    print("Building MPI layers...")
    N = 4
    layers_rgba, plane_centers = build_mpi_layers(left, disp_norm, num_planes=N, soft_sigma=0.07)

    # 4) inpaint holes per layer (optional)
    print("Inpainting layers...")
    layers_filled = [inpaint_layer_alpha(rgba) for rgba in layers_rgba]

    # 5) save layers
    save_layers(layers_filled, out_dir="mpi_layers")

    # 6) composite back to check (should resemble left)
    print("Compositing layers to validate reconstruction...")
    recon = composite_layers(layers_filled)
    cv2.imwrite("reconstruction.png", recon)

    # 7) render a few intermediate views
    print("Rendering views...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out = render_view_from_mpi(layers_filled, plane_centers, disp_norm, t=t, max_shift_px=24)
        cv2.imwrite(f"render_t_{int(t*100)}.png", out)

    print("Done. Outputs: mpi_layers/, reconstruction.png, render_t_*.png")

if __name__ == "__main__":
    main()
