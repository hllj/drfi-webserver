from feature_process import Features
from region_detect import Super_Region
import numpy as np

from .show_segmentation_level import show_segmentation_level

C_LIST = [20, 80, 350, 900]


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
            similarity[i, ids] = rf.predict(X)[:, 0]
        for idx, c in enumerate(C_LIST):
            rlist, rmat = Super_Region.combine_region(
                similarity, c, self.rlist, self.rmat
            )
            show_segmentation_level(idx + 1, rlist, self.img_path)
            if len(rlist) == 1:
                rlist = self.rlists[-1].copy()
                rmat = self.rmats[-1].copy()
            self.rlists.append(rlist)
            self.rmats.append(rmat)
            features = Features(self.img_path, rlist, rmat, need_comb_features=False)
            self.feature93s.append(features.features93)
