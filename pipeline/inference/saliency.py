import cv2
import numpy as np
from .base import InferenceBase


class Saliency(InferenceBase):
    def inference(self, img_data, similarities):
        img_data.get_multi_segs(similarities)
        saliency_model = self.get_model()
        segs_num = len(img_data.rlists)
        height = img_data.rmat.shape[0]
        width = img_data.rmat.shape[1]
        salience_map = np.zeros([segs_num, height, width])
        for i, rlist in enumerate(img_data.rlists):
            if len(rlist) == 1:
                print("Warning: Only have 1 region")
                for j, r in enumerate(rlist):
                    salience_map[i][r] = 0.5
                continue
            Y = saliency_model.predict(img_data.feature93s[i])[:, 1]
            for j, r in enumerate(rlist):
                salience_map[i][r] = Y[j]
            salience_img = salience_map[i, :, :].copy()
            salience_img = (salience_img - salience_img.min()) / (
                salience_img.max() - salience_img.min()
            )
            salience_img *= 255
            salience_img = salience_img.astype(np.uint8)
            cv2.imwrite("Saliency level {}.jpg".format(i), salience_img)
        X = salience_map.reshape([-1, height * width]).T
        return X
