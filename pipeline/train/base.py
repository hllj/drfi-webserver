class TrainingBase:
    def __init__(
        self,
        list_img_path: list,
        list_seg_path: list,
        list_img_level0: list,
        list_img_data: list,
        model_path,
        model_type="XGB",
    ):
        self.img_path = list_img_path
        self.img_data_level0 = list_img_level0
        self.seg_path = list_seg_path
        self.img_data = list_img_data
        self.model_type = model_type
        self.model_path = model_path

    def execute(self, *args, **kwargs):
        pass

    def prepare_data(self, data):
        return data[:, 0], data[:, 1:]
