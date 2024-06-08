import cv2
import matplotlib.pyplot as plt
from filters import gaussian, median, bilateral, sobel

img_path = 'image/IMG_2429.JPG'
method = bilateral

# 画像の読み込み（カラー）
image = cv2.imread('')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 任意フィルタの適用
smoothed_image = method(image)

# 結果の表示
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(image), plt.title('Original Image')
plt.subplot(1,2,2), plt.imshow(smoothed_image), plt.title('Filtered Image')
plt.show()
