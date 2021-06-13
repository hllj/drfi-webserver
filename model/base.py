class BaseModel:
    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        raise NotImplementedError()

    def test(self, X_test, Y_test):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def load_model(self, model_path):
        raise NotImplementedError()

    def save_model(self, model_path):
        raise NotImplementedError()
