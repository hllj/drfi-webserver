import numpy as np
import os
import click
import argparse
import pickle
import itertools
import cv2
import json


from measures.score import get_F_measure, get_MAE
from measures.custom_roc_curve import custom_roc_curve
from sklearn import metrics


def get_bounding_box(mask):
    white_indices = np.where(mask == [255])
    white_coordinates = tuple(zip(white_indices[1], white_indices[0]))
    x, y, w, h = cv2.boundingRect(np.array(white_coordinates))
    box = (x, y, x + w, y + h)
    return box


def grabcut_background(img, x1, y1, x2, y2):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros(img.shape[:2], np.uint8)
    box = (x1, y1, x2, y2)
    mask, bgdModel, fgdModel = cv2.grabCut(
        img, mask, box, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT
    )
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return mask2[:, :, np.newaxis]


def grabcut_prob(img, mask):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(
        img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
    )
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return mask[:, :, np.newaxis]


def grabcut_inference(img, Y):
    Y = (Y - Y.min()) / Y.max() * 255
    Y = Y.astype("uint8")
    mask_bb = Y.copy()
    mask_bb = np.where(mask_bb >= Y.mean(), 255, 0)
    mask_pr = Y.copy()

    mask_pr[Y <= 10] = cv2.GC_BGD
    mask_pr[(Y > 10) & (Y < 127)] = cv2.GC_PR_BGD
    mask_pr[(Y >= 127) & (Y < 255)] = cv2.GC_PR_FGD
    mask_pr[Y == 255] = cv2.GC_FGD

    (x1, y1, x2, y2) = get_bounding_box(mask_bb)

    # mask1 = grabcut_background(img, x1, y1, x2, y2)
    # img *= mask1
    mask2 = grabcut_prob(img, mask_pr)

    mask = mask2
    return mask * 255


@click.command()
@click.option("--config_path", default="./config/config.yaml")
@click.option("--project_folder", default="./")
def main(config_path, project_folder):
    its = [i for i in range(0, 5000) if (i % 5 == 0) and (i != 80)]
    ROOT_FOLDER = os.getcwd()
    PROJECT_FOLDER = os.path.join(ROOT_FOLDER, project_folder)
    img_paths = [
        os.path.join(PROJECT_FOLDER, "data/MSRA-B/{}.jpg".format(i)) for i in its
    ]
    seg_paths = [
        os.path.join(PROJECT_FOLDER, "data/MSRA-B/{}.png".format(i)) for i in its
    ]
    predict_paths = [
        os.path.join(PROJECT_FOLDER, "data/predict/{}.jpg".format(i)) for i in its
    ]
    grabcut_paths = [
        os.path.join(PROJECT_FOLDER, "data/grabcut/{}.jpg".format(i)) for i in its
    ]
    test_its = its[:100]
    test_img_paths = img_paths[:100]
    test_seg_paths = seg_paths[:100:]
    test_predict_paths = predict_paths[:100]
    test_grabcut_paths = grabcut_paths[:100]
    custom_threshold = np.arange(0, 1, 0.001)
    total_fpr = np.zeros_like(custom_threshold)
    total_tpr = np.zeros_like(custom_threshold)
    total_tp = 0.0
    total_fp = 0.0
    total_tn = 0.0
    total_fn = 0.0
    MAE = 0.0
    cnt = 0
    for i in range(len(test_its)):
        if os.path.exists(test_predict_paths[i]) is False:
            continue
        print("Img:", test_img_paths[i])
        img = cv2.imread(test_img_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Y = cv2.imread(test_predict_paths[i], cv2.IMREAD_GRAYSCALE)
        mask_bb = Y.copy()
        mask_bb = np.where(mask_bb >= Y.mean(), 255, 0)
        mask_pr = Y.copy()

        mask_pr[Y <= 20] = cv2.GC_BGD
        mask_pr[(Y > 20) & (Y < 200)] = cv2.GC_PR_BGD
        mask_pr[(Y >= 200) & (Y < 255)] = cv2.GC_PR_FGD
        mask_pr[Y >= 255] = cv2.GC_FGD

        (x1, y1, x2, y2) = get_bounding_box(mask_bb)

        mask1 = grabcut_background(img, x1, y1, x2, y2)
        # img *= mask1
        mask2 = grabcut_prob(img, mask_pr)

        mask = mask2
        ground_truth = cv2.imread(test_seg_paths[i])[:, :, 0]
        ground_truth[ground_truth == 255] = 1
        fpr, tpr = custom_roc_curve(
            ground_truth.reshape(-1), mask.reshape(-1), custom_threshold
        )
        total_fpr += fpr
        total_tpr += tpr
        fp = np.sum((mask.reshape(-1) == 1) & (ground_truth.reshape(-1) == 0))
        tp = np.sum((mask.reshape(-1) == 1) & (ground_truth.reshape(-1) == 1))

        fn = np.sum((mask.reshape(-1) == 0) & (ground_truth.reshape(-1) == 1))
        tn = np.sum((mask.reshape(-1) == 0) & (ground_truth.reshape(-1) == 0))

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        MAE += get_MAE(mask.reshape(-1), ground_truth.reshape(-1))
        cnt += 1
        cv2.imwrite(test_grabcut_paths[i], mask * 255)

    total_fpr /= cnt
    total_tpr /= cnt
    auc = metrics.auc(total_fpr, total_tpr)
    print("Calculate avg auc using grabcut", auc)
    if total_tp + total_fp == 0:
        precision = 1.0
    else:
        precision = total_tp / (total_tp + total_fp)

    if total_tp + total_fn == 0:
        recall = 1.0
    else:
        recall = total_tp / (total_tp + total_fn)
    f_measure = get_F_measure(precision, recall)
    MAE /= cnt
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measure:", f_measure)
    print("MAE:", MAE)


if __name__ == "__main__":
    main()
