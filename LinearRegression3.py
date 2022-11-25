import numpy as np


def get_NSE_Score(yTest, yPredict):
    return 1 - np.sum((yTest - yPredict) ** 2) / np.sum((yTest - np.mean(yTest)) ** 2)


def get_R2_Score(yTest, yPredict):
    return (
        np.sum((yTest - np.mean(yTest)) * (yPredict - np.mean(yPredict)))
        / np.sqrt(
            np.sum((yTest - np.mean(yTest)) ** 2)
            * np.sum((yPredict - np.mean(yPredict)) ** 2)
        )
    ) ** 2


def get_MAE_Score(yTest, yPredict):
    return np.sum(np.abs(yTest - yPredict)) / len(yTest)


def get_RMSE_Score(yTest, yPredict):
    return np.sqrt(np.sum((yTest - yPredict) ** 2) / len(yTest))
