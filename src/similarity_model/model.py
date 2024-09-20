from sklearn.base import BaseEstimator
import torch


def angle_similarity(a, b):
    return torch.acos(torch.nn.functional.cosine_similarity(a, b))


class AngleSimilarityModel(BaseEstimator):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        print(f"{X=}")
        self.model = angle_similarity(X[:, 800], X[:, 800:])

    def predict(self, X):
        return self.model

    def score(self, X, y):
        return self.model
