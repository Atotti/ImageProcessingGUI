import cv2

def bilateral(image, d=9, sigmaColor=75, sigmaSpace=75, count=10):
    smoothed_image = image
    for i in range(count):
        smoothed_image = cv2.bilateralFilter(smoothed_image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return smoothed_image

def gaussian(image, ksize=5, sigmaX=0):
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX)

def median(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def sobel(image, ksize=3):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

