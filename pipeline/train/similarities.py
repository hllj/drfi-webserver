from .base import TrainingBase
from utils.img_data import Img_Data
from region_detect import Super_Region, Region2Csv
import pickle
import os
import numpy as np
import logging
from model.model_factory import ModelFactory


class Similarities(TrainingBase):
    def execute(self):
        model = ModelFactory.get_model(model_name=self.model_type)
        if os.path.isfile(self.model_path):
            model.load_model(self.model_path)
            return model

        simi_data = None
        for i, path in enumerate(self.img_path):
            im_data = self.check_exist(
                img_path=self.img_path[i], img_path_level0=self.img_data_level0[i]
            )

            data = Region2Csv.generate_similar_csv(
                im_data.rlist, im_data.comb_features, self.seg_path[i]
            )
            if simi_data is None:
                simi_data = data
            else:
                simi_data = np.vstack((simi_data, data))
            logging.info("Finished simi {}".format(i))

        y_train, x_train = self.prepare_data(simi_data)

        model.train(x_train, y_train)
        model.save_model(self.model_path)
        return model

    def check_exist(self, img_path_level0, img_path) -> Img_Data:
        if os.path.exists(img_path_level0) is True:
            im_data = pickle.load(open(img_path_level0, "rb+"))
        else:
            im_data = Img_Data(img_path)
            with open(img_path_level0, "wb+") as f:
                pickle.dump(im_data, f)

        return im_data
