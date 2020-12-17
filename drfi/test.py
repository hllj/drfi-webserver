import cv2
import numpy as np

import argparse
import os


from drfi.model import RandomForest, MLP, XGB
from drfi.feature_process import Features
from drfi.region_detect import Super_Region, Region2Csv

C_LIST = [20, 80, 350, 900]
ROOT_FOLDER = os.getcwd()


def show_segmentation_level(level, rlist, img_path):
    img = cv2.imread(img_path)
    seg_img = np.zeros_like(img)
    for plist in rlist:
        r_color = list(np.random.choice(range(256), size=3))
        color = [int(r_color[0]), int(r_color[1]), int(r_color[2])]
        _x, _y = plist
        seg_img[_x, _y] = color
    # cv2.imshow("level {}".format(level), seg_img)


class Img_Data:
    def __init__(self, img_path):
        self.img_path = img_path
        self.rlist, self.rmat = Super_Region.get_region(img_path, 100.0)
        show_segmentation_level(0, self.rlist, self.img_path)
        features = Features(img_path, self.rlist, self.rmat)
        self.comb_features = features.comb_features
        self.rlists = [self.rlist]
        self.rmats = [self.rmat]
        self.feature93s = [features.features93]

    def get_multi_segs(self, rf):
        num_reg = len(self.rlist)
        similarity = np.ones([num_reg, num_reg])
        for i in range(num_reg):
            ids = self.comb_features[i]["j_ids"]
            X = self.comb_features[i]["features"]
            similarity[i, ids] = 1 - rf.predict(X)[:, 1]
        for idx, c in enumerate(C_LIST):
            rlist, rmat = Super_Region.combine_region(
                similarity, c, self.rlist, self.rmat
            )
            show_segmentation_level(idx, rlist, self.img_path)
            if len(rlist) == 1:
                continue
            self.rlists.append(rlist)
            self.rmats.append(rmat)
            features = Features(self.img_path, rlist, rmat, need_comb_features=False)
            self.feature93s.append(features.features93)


def get_transparent_image(img_path, height, width, Y, threshold):
    img = np.zeros([height, width * 2, 3], dtype=np.uint8)
    original_img = cv2.imread(img_path)
    img[:, :width, :] = original_img
    mask = Y.repeat(3).reshape([height, width, 3])
    img[:, width:, :] = mask
    # cv2.imshow("Result", img)
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


def get_fusion_model(fm, model_path_fusion):
    model = None
    if fm == "mlp":
        model = MLP()
    elif fm == "xgb":
        model = XGB()
    model.load_model(model_path_fusion)
    return model


def main(
    img_path,
    threshold,
    fm,
    model_path_same_region,
    model_path_salience,
    model_path_fusion,
):

    RESULT_FOLDER = os.path.join(ROOT_FOLDER, "drfi/data/result/")
    im_data = Img_Data(img_path)

    rf_simi = RandomForest()
    rf_simi.load_model(model_path_same_region)
    rf_sal = RandomForest()
    rf_sal.load_model(model_path_salience)

    im_data.get_multi_segs(rf_simi)
    segs_num = len(im_data.rlists)
    height = im_data.rmat.shape[0]
    width = im_data.rmat.shape[1]
    salience_map = np.zeros([segs_num, height, width])
    for i, rlist in enumerate(im_data.rlists):
        Y = rf_sal.predict(im_data.feature93s[i])[:, 1]
        for j, r in enumerate(rlist):
            salience_map[i][r] = Y[j]
    X_test = salience_map.reshape([-1, height * width]).T
    fusion_model = get_fusion_model(fm, model_path_fusion)
    Y = fusion_model.predict(X_test).reshape([height, width]) * 255

    print("finished~( •̀ ω •́ )y")
    result_img = get_transparent_image(img_path, height, width, Y, threshold)
    if os.path.isdir(RESULT_FOLDER) is False:
        os.mkdir(RESULT_FOLDER)
    img_name = img_path.split("/")[-1].split(".")[0]
    result_filename = img_name + ".png"
    result_path = RESULT_FOLDER + result_filename
    cv2.imwrite(result_path, result_img)
    return result_filename, result_path, result_img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to test image directory")
    ap.add_argument("-t", "--threshold", required=True, help="hard threshold")
    ap.add_argument(
        "-fm",
        "--fusionmodel",
        nargs="?",
        help="select model for fusion stage",
        const="mlp",
    )
    ap.add_argument(
        "-mpsr",
        "--modelpathsameregion",
        required=True,
        help="path to model predicting same region directory",
    )
    ap.add_argument(
        "-mps",
        "--modelpathsalience",
        required=True,
        help="path to model predict salience directory",
    )
    ap.add_argument(
        "-mpf",
        "--modelpathfusion",
        required=True,
        help="path to model fusion directory",
    )
    args = vars(ap.parse_args())

    img_path = args["image"]
    threshold = int(args["threshold"])
    fm = args["fusionmodel"]
    model_path_same_region = args["modelpathsameregion"]
    model_path_salience = args["modelpathsalience"]
    model_path_fusion = args["modelpathfusion"]

    RESULT_FOLDER = os.path.join(ROOT_FOLDER, "data/result/")
    im_data = Img_Data(img_path)

    rf_simi = RandomForest()
    # model_path = "data/model/rf_same_region.pkl"
    rf_simi.load_model(model_path_same_region)
    rf_sal = RandomForest()
    # model_path = "data/model/rf_salience.pkl"
    rf_sal.load_model(model_path_salience)

    im_data.get_multi_segs(rf_simi)
    segs_num = len(im_data.rlists)
    height = im_data.rmat.shape[0]
    width = im_data.rmat.shape[1]
    salience_map = np.zeros([segs_num, height, width])
    for i, rlist in enumerate(im_data.rlists):
        Y = rf_sal.predict(im_data.feature93s[i])[:, 1]
        for j, r in enumerate(rlist):
            salience_map[i][r] = Y[j]
    X_test = salience_map.reshape([-1, height * width]).T

    fusion_model = get_fusion_model(fm, model_path_fusion)
    Y = fusion_model.predict(X_test).reshape([height, width]) * 255

    print("finished~( •̀ ω •́ )y")
    result_img = get_transparent_image(img_path, height, width, Y, threshold)
    if os.path.isdir(RESULT_FOLDER) is False:
        os.mkdir(RESULT_FOLDER)
    img_name = args["image"].split("/")[-1].split(".")[0]
    result_path = RESULT_FOLDER + img_name + ".png"
    cv2.imwrite(result_path, result_img)
    cv2.waitKey(0)
