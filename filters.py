import cv2
import numpy as np


def bilateral(image, d=9, sigmaColor=75, sigmaSpace=75, count=10):
    smoothed_image = image
    for i in range(count):
        smoothed_image = cv2.bilateralFilter(
            smoothed_image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
        )
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


def low_pass_filter(image, cutoff_frequency=100):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a mask first, center square is 1, remaining all zeros
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[
        crow - cutoff_frequency : crow + cutoff_frequency,
        ccol - cutoff_frequency : ccol + cutoff_frequency,
    ] = 1

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize to 8-bit range
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # Convert grayscale back to BGR
    img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

    return img_back


def high_pass_filter(image, cutoff_frequency=30):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a mask first, center square is 0, remaining all ones
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[
        crow - cutoff_frequency : crow + cutoff_frequency,
        ccol - cutoff_frequency : ccol + cutoff_frequency,
    ] = 0

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize to 8-bit range
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # Convert grayscale back to BGR
    img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

    return img_back
