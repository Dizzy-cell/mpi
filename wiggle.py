"""
wiggle_from_depth.py
生成 wiggle 特效（左右摆动）——输入：left image + depth map (0..1)
依赖: opencv-python, numpy, imageio
pip install opencv-python numpy imageio
"""

import cv2
import numpy as np
import imageio
import os
from math import sin, cos, pi

# ---------- 配置 ----------
LEFT_IMG = "frame_1.png"   # 左图
DEPTH_INPUT = "depth.png"     # 深度可为 .npy (float32 0..1) 或 单通道图片 (.png,.jpg)
OUT_GIF = "wiggle_cat.gif"
FRAMES = 24       # 单次循环帧数（从 left->right）
CYCLES = 2        # 循环次数
MAX_SHIFT_PCT = 0.02  # 最大平移比例（相对于图像宽度），例如 0.02 = 2%
INPAINT_RADIUS = 3
USE_SIN = True     # 用 sin 平滑插值 (更顺滑)
HOLE_FILL_METHOD = "inpaint"  # "inpaint" 或 "dilate" 或 "none"
# ------------------------

def load_depth(path, target_shape=None, blur_size=5, mode='dilate', kernel_size=3):
    if path.lower().endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d_img is None:
            raise RuntimeError("Cannot read depth: " + path)
        if d_img.ndim == 3:
            d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
        # normalize to 0..1
        d = d_img.astype(np.float32)
        d -= d.min()
        if d.max() > 1e-8:
            d /= d.max()
    if target_shape is not None and (d.shape[1], d.shape[0]) != target_shape:
        d = cv2.resize(d, target_shape, interpolation=cv2.INTER_LINEAR)
    
        # 膨胀 or 腐蚀
    if mode == 'dilate':
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        d = cv2.dilate(d, kernel, iterations=15)
    elif mode == 'erode':
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        d = cv2.erode(d, kernel, iterations=15)
    
    # 高斯模糊，让层之间更平滑（防止分层断裂）
    d = cv2.GaussianBlur(d, (blur_size, blur_size), sigmaX=0)
    cv2.imwrite("depth_blur.png", (d / np.max(d) * 255).astype(np.uint8))
    
    return d

def generate_shift_map(depth, max_shift_px, invert_depth=False):
    # depth: HxW in 0..1, by default 0=near, 1=far. adjust invert_depth accordingly.
    if invert_depth:
        depth = 1.0 - depth
    # compute shift per-pixel (near -> big shift)
    # here we map near (0) => max_shift, far(1)=>0
    shift = (1.0 - depth) * max_shift_px
    return shift.astype(np.float32)

def warp_with_shift(image, shift_map, t):
    """
    inverse mapping remap:
      For output pixel (x,y), sample from src (x - t*shift/2, y)
    t in [-1,1]
    """
    H, W = shift_map.shape
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    # shift factor: fraction of full shift. t in [-1,1]
    map_x = (grid_x - (t * shift_map / 2.0)).astype(np.float32)
    map_y = grid_y.astype(np.float32)
    # remap with border black
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

def detect_holes(img):
    # holes where all channels == 0
    if img.ndim == 2:
        mask = (img == 0).astype(np.uint8) * 255
    else:
        mask = np.all(img == 0, axis=2).astype(np.uint8) * 255
    # optionally morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def fill_holes(img, mask, method="inpaint"):
    if method == "none":
        return img
    if mask.sum() == 0:
        return img
    if method == "dilate":
        # simple depth-guided color dilation: expand neighbor pixels into holes
        kernel = np.ones((3,3), np.uint8)
        filled = img.copy()
        for i in range(3):
            channel = filled[:,:,i]
            channel_masked = channel.copy()
            # repeatedly dilate until holes filled (capped iterations)
            for _ in range(4):
                channel_masked = cv2.dilate(channel_masked, kernel, iterations=1)
                channel[mask==255] = channel_masked[mask==255]
            filled[:,:,i] = channel
        return filled
    # default: inpaint
    # cv2.inpaint needs 8-bit 3-channel BGR and mask (255)
    img_8 = img.copy()
    if img_8.dtype != np.uint8:
        img_8 = np.clip(img_8*255.0,0,255).astype(np.uint8)
    inpainted = cv2.inpaint(img_8, mask, INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
    return inpainted

def make_wiggle_frames(left_img, depth_map, frames=24, cycles=1, max_shift_pct=0.02):
    H, W = left_img.shape[:2]
    max_shift_px = max(1, int(W * max_shift_pct))
    shift_map = generate_shift_map(depth_map, max_shift_px, invert_depth=False)
    out_frames = []
    total = frames * cycles
    # create t sequence: -1 -> 1 -> -1 (smooth)
    for cycle in range(cycles):
        for i in range(frames):
            if USE_SIN:
                # map i in [0,frames-1] to angle 0..pi, then to -1..1
                angle = (i / (frames - 1)) * pi
                t = -1.0 + 2.0 * (0.5 * (1 - cos(angle))) if False else -1.0 + 2.0 * (0.5*(1 - cos(angle))) 
                # simpler: sin-based smooth
                t = sin((i / (frames - 1)) * pi * 0.5) * 1.0  # 0..1
                t = -1.0 + 2.0 * ( (i) / (frames-1) )  # fallback linear
                # Better: use sine ease-in-out across full -1..1
                t = -1.0 * 1.0 + ( (1 + sin( (i/ (frames-1) - 0.5) * pi )) )  # just keep simple below
                # OK let's use a clearer smooth: s = 0.5*(1 - cos(pi * i/(frames-1))) => 0..1; t = -1 + 2*s
                s = 0.5 * (1 - np.cos(np.pi * i / (frames - 1)))
                t = -1.0 + 2.0 * s
            else:
                t = -1.0 + 2.0 * (i / (frames - 1))
            # warp
            warped = warp_with_shift(left_img, shift_map, t)
            # detect holes and fill
            mask = detect_holes(warped)
            if HOLE_FILL_METHOD != "none":
                filled = fill_holes(warped, mask, method=HOLE_FILL_METHOD)
            else:
                filled = warped
            out_frames.append(filled)
    return out_frames

def save_gif(frames, out_path, fps=24):
    # imageio expects RGB
    imgs = []
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f*255.0,0,255).astype(np.uint8)
        # cv2 uses BGR; convert to RGB for gif
        if f.ndim == 3 and f.shape[2] == 3:
            fr = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        else:
            fr = f
        imgs.append(fr)
    imageio.mimsave(out_path, imgs, fps=fps)
    print("Saved", out_path)

def save_video(frames, out_path, fps=24):
    # 确保至少有一帧
    if len(frames) == 0:
        print("No frames to save!")
        return
    
    # 取第一帧确定分辨率
    h, w = frames[0].shape[:2]
    
    # 定义编码器与输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'avc1'、'H264'
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        # 保证是 BGR 格式（OpenCV 写视频要求）
        if f.ndim == 3 and f.shape[2] == 3:
            frame = f
        else:
            frame = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    
    out.release()
    print("Saved video:", out_path)

def save_video_v2(frames, out_path, fps=24):
    frames_rgb = []
    for f in frames:
        frames_rgb.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    
    pred_mp4 = np.stack(frames_rgb, axis=0)  # T x H x W x 3
    imageio.mimwrite(out_path, pred_mp4, fps=fps
                     codes='libx264',
                     quality=8,)

def main():
    left = cv2.imread(LEFT_IMG, cv2.IMREAD_COLOR)
    if left is None:
        raise RuntimeError("Cannot read left image: " + LEFT_IMG)
    depth = load_depth(DEPTH_INPUT, target_shape=(left.shape[1], left.shape[0]))
    print("Loaded left:", left.shape, "depth:", depth.shape)
    frames = make_wiggle_frames(left, depth, frames=FRAMES, cycles=CYCLES, max_shift_pct=MAX_SHIFT_PCT)
    #save_gif(frames, OUT_GIF, fps=12)
    #save_video(frames, OUT_GIF.replace('.gif','.mp4'), fps=12)
    save_video_v2(frames, OUT_GIF.replace('.gif','.mp4'), fps=12)
if __name__ == "__main__":
    main()
