import numpy as np
import pandas as pd
from typing import Any, List, Dict, Collection, Union, Callable


""" 
    Aggregation Functions:
        np.ndarray[float]-> float
"""


def simple_average(predictions: np.ndarray, **kwargs) -> np.ndarray:
    return np.average(predictions, axis=0)

        #return predictions.mean()


def score_positive_average(predictions: np.ndarray, _self, **kwargs) -> float:
    return predictions[_self.scores > 0].mean()


def top_average(predictions: np.ndarray, _self, n_pct, **kwargs) -> np.ndarray:#取自己需要的比较好的百分比的部分进行取平均
    theta = np.percentile(_self.scores, n_pct)
    #return predictions[_self.scores > theta].mean()
    return np.average(predictions[_self.scores >= theta], axis=0)

