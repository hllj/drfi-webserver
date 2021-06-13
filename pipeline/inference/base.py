from model.model_factory import ModelFactory


class InferenceBase:
    def __init__(self, model_path, model_type="XGB"):
        self.model_path = model_path
        self.model_type = model_type

    def get_model(self):
        model = ModelFactory.get_model(self.model_type)
        model.load_model(self.model_path)
        return model

    def inference(self, *args, **kwargs):
        pass
