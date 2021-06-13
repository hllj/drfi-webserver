class Config:
    def __init__(self, args: dict):
        self.train_size = args.get("train_size")
        self.project_folder = args.get("project_folder")
        path = args.get("model_path")
        self._same_region = path[0].get("same_region")
        self._salience = path[1].get("salience")
        self._fusion = path[2].get("fusion")
        self._fusion_model_type = args.get("fusion_model_type")

        # for inference
        inference = args.get("inference")
        self._inference_img_path = inference[0].get("img_path")
        self._inference_threshold = inference[1].get("threshold")
        self._inference_output = inference[2].get("output")

    def get_same_region_path(self):
        return self._same_region

    def get_salience_path(self):
        return self._salience

    def get_fusion_path(self):
        return self._fusion

    def get_fusion_model_type(self):
        return self._fusion_model_type

    def get_inference_img_path(self):
        return self._inference_img_path

    def get_inference_threshold(self):
        return self._inference_threshold

    def get_inference_output(self):
        return self._inference_output
