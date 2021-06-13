import os
import pickle
import cv2

import numpy as np

from model.model_factory import ModelFactory
from .base import TrainingBase


class Fusion(TrainingBase):
    def __init__(
        self,
        list_img_path: list,
        list_seg_path: list,
        list_img_level0: list,
        list_img_data: list,
        model_path,
        C_LIST: list,
        model_type="XGB",
    ):
        super().__init__(
            list_img_path,
            list_seg_path,
            list_img_level0,
            list_img_data,
            model_path,
            model_type,
        )
        self.C_LIST = C_LIST

    def execute(self, model_sal):
        model = ModelFactory.get_model(self.model_type)
        if os.path.isfile(self.model_path):
            model.load_model(self.model_path)
            return model
        ground_truths = None
        salience_maps = None

        for i, path in enumerate(self.img_data):
            im_data = pickle.load(open(path, "rb+"))
            seg_num = len(im_data.rlists)
            if seg_num < len(self.C_LIST) + 1:
                continue

            height = im_data.rmat.shape[0]
            width = im_data.rmat.shape[1]
            salience_map = np.zeros([seg_num, height, width])
            for j, rlist in enumerate(im_data.rlists):
                Y = model_sal.predict(im_data.feature93s[j])[:, 1]
                for k, r in enumerate(rlist):
                    salience_map[j][r] = Y[k]
            ground_truth = cv2.imread(self.seg_path[i])[:, :, 0]
            ground_truth[ground_truth == 255] = 1
            if salience_maps is None:
                salience_maps = salience_map.reshape([-1, height * width]).T
            else:
                salience_maps = np.append(
                    salience_maps, salience_map.reshape([-1, height * width]).T, axis=0
                )
            if ground_truths is None:
                ground_truths = ground_truth.reshape(-1)
            else:
                ground_truths = np.append(
                    ground_truths, ground_truth.reshape(-1), axis=0
                )
        x_train = salience_maps
        y_train = ground_truths
        model.train(x_train, y_train)
        model.save_model(self.model_path)
