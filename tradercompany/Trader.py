import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Collection, Union, Callable
warnings.filterwarnings("ignore")
from statsmodels.api import OLS

from .activations import *
from .binaryops import *
from .Formula import Formula


class Trader:

    def __init__(self, weights, formulas, max_lag):
        self.n_terms = len(formulas)
        self.weights = weights
        self.formulas = formulas
        self.max_lag = max_lag
        self.score = 0
        self._pred_hist = np.zeros(1)*np.nan  # length T
        self._pred_hist_formulas = np.zeros([1, len(formulas)])*np.nan  # shape: [T, n_terms]
        self._time_index = 0




    def predict(self, feature_arr: np.ndarray) -> float:
        """ traderの予測値を返す
        Args:
            feature_array (np.ndarray): array with shape of [time, #feature]
        Returns:
            float: predict value for targeting asset
        """
        return sum([self.weights[j] * self.formulas[j].predict(feature_arr) for j in range(self.n_terms)])


    def _predict_with_formula(self, feature_arr: np.ndarray) -> List:
        """
        计算并返回交易者的预测值和各formula的预测值。
        Args:
            feature_arr (np.ndarray):
        Returns:
            List[float, np.ndarray[float]]: [pred(trader), [pred(formula_i)]]
        """
        pred_formulas = np.array([self.formulas[j].predict(feature_arr) for j in range(self.n_terms)])  # (1, #terms)
        pred_trader = np.sum(self.weights * pred_formulas)
        return pred_trader, pred_formulas.reshape(1, -1)


    def _recalc_predicts_hist(self, feature_arr_block: np.ndarray) -> None:
        """ 重置预测值的历史。全部重新计算。
        Args:
            feature_arr_block (np.ndarary): array with shape of [time, maxlag, #feature]
        Effects:
            self._pred_hist
            self._pred_hist_formulas
        """
        T = feature_arr_block.shape[0]
        self._time_index = T
        self._pred_hist = np.zeros(T)
        self._pred_hist_formulas = np.zeros([T, self.n_terms])
        for t in range(T):
            pred, preds_f = self._predict_with_formula(feature_arr_block[t])  # maxlag=0の場合1行目を取り出す.
            self._pred_hist[t] = pred
            self._pred_hist_formulas[t] = preds_f


    def _update_score(self, return_arr: np.ndarray, eval_lookback: int, eval_method="default") -> None:
        """ 
        Args:
            return_arr (np.ndarray): 
            eval_method (str, optional): 
                which method to use to evaluate traders.
                - "default":
                    cumulative return : sum sign(pred_t) * ret_t
        Effects:
            self.score (positive is good)
        """
        if eval_method == "default":
            self.score = np.sum( (np.sign(self._pred_hist)[-eval_lookback:] * return_arr[-eval_lookback:]) )
#sign 大于0等于1 小于0等于-1 等于0等于0
        else:
            raise NotImplementedError(f"unknown eval_method : {eval_method}")


    def recalc(self, feature_arr_block: np.ndarray, return_arr: np.ndarray, eval_lookback, eval_method="default"):
        """ 予測値の履歴をリセット計算し直す. 合わせて評価値も計算し直す.
        Args:
            feature_arr_block (np.ndarray):
            return_arr (np.ndarrayeval_method, optional): [description]. Defaults to "default".
        Returns:
            Trader: self
        """
        self._recalc_predicts_hist(feature_arr_block)
        self._update_score(return_arr, eval_lookback, eval_method)
        return self


    def _append_predicts_hist(self, feature_arr: np.ndarray) -> None:
        """ トレーダおよび保持しているFormulaの予測値履歴の末尾を追記.
        跟踪器和持有的Formula预测值历史的末尾。
        Args:
            feature_arr (np.ndarray):
        Effects:
            self._pred_hist
            self._pred_hist_formulas
        """
        self._time_index += 1
        pred, preds_f = self._predict_with_formula(feature_arr)
        self._pred_hist = np.append(self._pred_hist, pred)
        self._pred_hist_formulas = np.append(self._pred_hist_formulas, preds_f, axis=0)


    def _update_weights(self, return_arr: np.ndarray, eval_lookback) -> None:
        """ weightsを更新
        Args:
            return_arr (np.ndarray):
        Effects:
            self.weights
        """
        y = return_arr[-eval_lookback:]
        X = self._pred_hist_formulas[-eval_lookback:]
        self.weights = OLS(y, X).fit().params


    def _to_numerical_repr(self) -> List:
        """
        Returns:
            List: {M, (formula_params)_j}
        """
        formula_arr = np.array([formula.to_numerical_repr() for formula in self.formulas])
        return self.n_terms, formula_arr


    def to_str(self, feature_names):
        return [str(round(w, 5))+", "+formula.to_str(feature_names) for w,formula in zip(self.weights, self.formulas)]


    def cumulative_pnl(self, return_arr: np.ndarray):
        assert return_arr.shape[0] == self._pred_hist.shape[0]
        return pd.Series(np.sign(self._pred_hist) * return_arr).cumsum()