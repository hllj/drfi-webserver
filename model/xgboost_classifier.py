import pickle
import xgboost as xgb
from xgboost import XGBClassifier

from .load_data import do_rebalance
from .base import BaseModel


class XGB(BaseModel):
    def __init__(self):
        self.clf = XGBClassifier(
            n_estimators=200,
            max_depth=20,
            learning_rate=0.1,
            random_state=0,
            booster="gbtree",
            use_label_encoder=False,
        )

    def train(self, X_train, Y_train):
        X_train, Y_train = do_rebalance(X_train, Y_train)
        self.clf.fit(X_train, Y_train)

    def test(self, X_test, Y_test):
        Y_prob = self.clf.predict_proba(X_test)
        auc = metrics.roc_auc_score(Y_test, Y_prob[:, 1])

    def predict(self, X):
        Y_prob = self.clf.predict_proba(X)
        return Y_prob

    def load_model(self, model_path):
        self.clf.load_model(model_path)
        # with open(model_path, "rb+") as file:
        #     self.clf = pickle.load(file)

    def save_model(self, model_path):
        self.clf.save_model(model_path)
        # with open(model_path, "wb+") as file:
        #     pickle.dump(self.clf, file)
