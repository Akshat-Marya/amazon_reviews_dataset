from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pickle
from model_xgboost import load_model

class ClfHelpfulness:

    def predict(self,X):
        """
        :param X: str
        """
        estimator=load_model()
        pred = estimator.predict([X])
        return pred