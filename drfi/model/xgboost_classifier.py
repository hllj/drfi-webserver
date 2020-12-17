import pickle
from xgboost import XGBClassifier
from sklearn import metrics
from .load_data import do_rebalance


class XGB:
    def __init__(self):
        self.clf = XGBClassifier(
            n_estimators=200,
            max_depth=20,
            learning_rate=0.1,
            booster="gbtree",
        )

    def train(self, X_train, Y_train):
        X_train, Y_train = do_rebalance(X_train, Y_train)
        self.clf.fit(X_train, Y_train)

    def test(self, X_test, Y_test):
        Y_prob = self.clf.predict_proba(X_test)
        auc = metrics.roc_auc_score(Y_test, Y_prob[:, 1])
        print("model's auc is {}".format(auc))

    def predict(self, X):
        Y_prob = self.clf.predict_proba(X)[:, 1]
        return Y_prob

    def load_model(self, model_path):
        with open(model_path, "rb+") as file:
            self.clf = pickle.load(file)

    def save_model(self, model_path):
        with open(model_path, "wb+") as file:
            pickle.dump(self.clf, file)
