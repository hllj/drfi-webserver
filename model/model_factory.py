from .base import BaseModel
from .multilayer_perceptron import MLP
from .random_forest import RandomForest
from .xgboost_classifier import XGB


class ModelFactory:
    __model = {}

    def register_model(self, model_name, model):
        self.__model[model_name] = model

    @staticmethod
    def get_model(model_name) -> BaseModel:
        model_obj = ModelFactory.__model.get(model_name)()
        if not model_obj:
            raise ValueError()
        return model_obj


model_factory = ModelFactory()
model_factory.register_model("MLP", MLP)
model_factory.register_model("RandomForest", RandomForest)
model_factory.register_model("XGB", XGB)
