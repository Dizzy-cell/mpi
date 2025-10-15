import cv2
#from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('cat_1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cv2.imwrite('cat_gray.png', image_gray)

depth_blur = cv2.GaussianBlur(image_gray, (5,5), 0)

bins = np.linspace(0, 255, num=32)
seg = np.digitize(depth_blur, bins)
cv2.imwrite('cat_seg.png', (seg / np.max(seg) * 255).astype(np.uint8))

for i in  range(1, seg.max()+1):
    mask = (seg == i)
    color = image * mask[:, :, None]
    cv2.imwrite(f'mpi_depth/cat_seg_{i}.png', color)

# segments = slic(image, n_segments=300, compactness=10, sigma=1)

# plt.imshow(mark_boundaries(image, segments))
# plt.axis('off')
# plt.show()
