import cv2
import numpy as np


def get_transparent_image(img, Y, threshold):
    original_img = img.copy()
    height = original_img.shape[0]
    width = original_img.shape[1]
    img = np.zeros([height, width * 2, 3], dtype=np.uint8)
    img[:, :width, :] = original_img
    mask = Y.repeat(3).reshape([height, width, 3])
    img[:, width:, :] = mask
    cv2.imwrite("Result.jpg", img)
    # get transparent image
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    mask = mask.astype(np.uint8)
    bitwise_img = cv2.bitwise_and(original_img, mask)
    bgra_img = cv2.cvtColor(bitwise_img, cv2.COLOR_BGR2BGRA)
    for j in range(height):
        for i in range(width):
            (b, g, r, a) = bgra_img[j, i]
            if (b == 0) and (g == 0) and (r == 0):
                bgra_img[j, i, 3] = 0
    return bgra_img
