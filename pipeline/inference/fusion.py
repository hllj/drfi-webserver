import numpy as np
from .base import InferenceBase


class Fusion(InferenceBase):
    def inference(self, X, im_data):
        height = im_data.rmat.shape[0]
        width = im_data.rmat.shape[1]
        fusion_model = self.get_model()
        print("loss:", fusion_model.clf.loss_)
        print(X.shape)
        Y = fusion_model.predict(X)[:, 1]
        Y = Y.reshape([height, width]) * 255
        return Y
