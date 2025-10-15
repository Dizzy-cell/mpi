
from pillow_heif import open_heif

path = "IMG_0802.HEIC"
path = "af50a19bc1cc776b993618dd4b04ea04.HEIC"
path = "cat-7976680_1280.heic"
heif_file = open_heif(path)

name = 'cat'

print(f"共包含 {len(heif_file._images)} 张图像")

for i, img in enumerate(heif_file._images):
    print(f"\n--- 图像层 {i+1} ---")
    print(f"尺寸: {img.size}")
    print(f"模式: {img.mode}")
    print(f"信息键: {list(img.info.keys())}")
    if "metadata" in img.info:
        print(f"元数据类型: {[m['type'] for m in img.info['metadata']]}")

# 如果你想导出图层：
from PIL import Image
for i, img in enumerate(heif_file._images):
    Image.frombytes(img.mode, img.size, img.data).save(f"{name}_{i+1}.png")
    print(f"导出 {name}_{i+1}.png")
