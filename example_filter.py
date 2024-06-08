import cv2
import matplotlib.pyplot as plt
from filters import gaussian, median, bilateral, sobel, low_pass_filter, high_pass_filter

img_path = 'image/IMG_2429.JPG'
method = low_pass_filter

# 画像の読み込み（カラー）
if method is low_pass_filter or method is high_pass_filter:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
else:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 任意フィルタの適用
filtered_image = method(image)

# 結果の表示
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(image), plt.title('Original Image')
plt.subplot(1,2,2), plt.imshow(filtered_image), plt.title('Filtered Image')
plt.show()
