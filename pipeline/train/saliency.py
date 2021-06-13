from .base import TrainingBase
from model.model_factory import ModelFactory
from utils.img_data import Img_Data
from region_detect import Region2Csv
from model.base import BaseModel
import os
import numpy as np
import pickle


class Saliency(TrainingBase):
    def execute(self, model_simi: BaseModel):
        model = ModelFactory.get_model(self.model_type)
        if os.path.isfile(self.model_path):
            model.load_model(self.model_path)
            return model

        sal_data = None

        for i, path in enumerate(self.img_path):
            im_data = self.parse_data(
                self.img_data[i], self.img_data_level0[i], model_simi
            )
            for j, rlist in enumerate(im_data.rlists):
                data = Region2Csv.generate_seg_csv(
                    rlist, im_data.feature93s[j], self.seg_path[i]
                )
                if data is None:
                    continue
                if sal_data is None:
                    sal_data = data
                else:
                    sal_data = np.vstack((sal_data, data))
        y_train, x_train = self.prepare_data(sal_data)

        model.train(x_train, y_train)
        model.save_model(self.model_path)
        return model

    def parse_data(self, img_data, img_data_level0, model_simi) -> Img_Data:
        im_data = pickle.load(open(img_data_level0, "rb+"))
        im_data.get_multi_segs(model_simi)
        with open(img_data, "wb+") as f:
            pickle.dump(im_data, f)
        return im_data
