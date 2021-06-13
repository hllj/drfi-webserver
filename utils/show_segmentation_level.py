import numpy as np
import cv2


def show_segmentation_level(level, rlist, img_path):
    img = cv2.imread(img_path)
    seg_img = np.zeros_like(img)
    for plist in rlist:
        r_color = list(np.random.choice(range(256), size=3))
        color = [int(r_color[0]), int(r_color[1]), int(r_color[2])]
        _x, _y = plist
        seg_img[_x, _y] = color
    cv2.imwrite("level {}.jpg".format(level), seg_img)