import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Collection, Union, Callable
warnings.filterwarnings("ignore")
from statsmodels.api import OLS
EXP_NAME = "/home/zwc/tutorial_exp"
from .activations import *
from .binaryops import *
from .Formula import Formula
from .Model import TransformerModel
import torch
from .transformer import PositionalEncoding,TransAm,AttnDecoder,StockDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
#from dataset import StockDataset
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch.autograd import Variable
#import argparse
import math
import torch.nn.functional as F
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import math
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

class myTrader:

    def __init__(self, d_feat,batch_size):
        #self.n_terms = len(formulas)
        #self.weights = weights
        #self.formulas = formulas
        #self.max_lag = max_lag
        self.train_loss=0
        self.stop_steps=0
        self.score = 0
        self.batch_size=batch_size
        self.T=0
        self.best_score=-np.inf
        self.train_loss=0
        self.train_score =-np.inf
        self.val_loss=0
        self.val_score=-np.inf
        #self._pred_hist = np.zeros(1)*np.nan  # length T
        self._pred_hist = np.zeros_like([[0] * batch_size] * 23)* np.nan
        #self._pred_hist_formulas = np.zeros([1, len(formulas)])*np.nan  # shape: [T, n_terms]
        self._time_index = 0
        self.Model = TransformerModel(d_feat=d_feat,batch_size=batch_size)

        self.best_param = self.Model
        #self.best_param = copy.deepcopy(self.Model.model.state_dict())
        # self.Model = Model(input_size=len(formulas),
        #                    hidden_layer_size=128,
        #                    output_size=1,)
        #                    #batch_size=16,
        #                    #lr=0.0001,
        #                    #cost='mean')
    def train_once(self,save_path):
        import pickle
        a_file = open(save_path, "wb")
        pickle.dump(self.best_param, a_file)
        a_file.close()
        self.Model.logger.info("best score: %.6lf" % (self.best_score))
        self.Model=self.best_param
        #self.Model.model.load_state_dict(self.best_param)

        if self.Model.use_gpu:
            torch.cuda.empty_cache()
    def refit(self):
        self.Model.logger.info("start...")
        self.Model.fitted = True
    def predict(self,  data):#####
        self.encoder.eval()
        self.decoder.eval()
        data_x = data.unsqueeze(2)
        data_tran = data_x.transpose(0, 1)
        data_x, data_y = data_tran.float(), data.float()
        code_hidden = self.encoder(data_x)
        #code_hidden = code_hidden.transpose(0, 1)
        #output = self.decoder(code_hidden, data_y)
        code_hidden = code_hidden.transpose(0, 1)
        output = self.decoder(code_hidden, data_y).squeeze(1)
        if len(output)<self.batch_size:
            return np.pad(output.detach().numpy(),(0,self.batch_size-len(output.detach().numpy())))
        else:
            return output.detach().numpy()
        """ traderの予測値を返す
        Args:
            feature_array (np.ndarray): array with shape of [time, #feature]
        Returns:
            float: predict value for targeting asset
        """
        #return sum([self.weights[j] * self.formulas[j].predict(feature_arr) for j in range(self.n_terms)])
        #return self.Model(torch.tensor(np.array([self.formulas[j].predict(feature_arr) for j in range(self.n_terms)]), dtype = torch.float)).detach().numpy()

    def _predict_with_formula(self, data_x,data_y) -> List:######
        """
        计算并返回交易者的预测值和各formula的预测值。
        Args:
            feature_arr (np.ndarray):
        Returns:
            List[float, np.ndarray[float]]: [pred(trader), [pred(formula_i)]]
        """
        #pred_value = torch.tensor(np.array([self.formulas[j].predict(data_x,data_y) for j in range(self.n_terms)]) , dtype = torch.float) # (1, #terms)
        pred_trader = self.predict(data_x,data_y)
        return pred_trader#, pred_formulas.reshape(1, -1)


    def _recalc_predicts_hist(self, data_x,data_y) -> None:#####
        """ 重置预测值的历史。全部重新计算。
        Args:
            feature_arr_block (np.ndarary): array with shape of [time, maxlag, #feature]
        Effects:
            self._pred_hist
            self._pred_hist_formulas
        """
        #T = feature_arr_block.shape[0]
        #self._time_index = T
        #self._pred_hist = np.zeros(T)
        #self._pred_hist_formulas = np.zeros([T, self.n_terms])
        pred = self.predict(data_x, data_y)  # maxlag=0の場合1行目を取り出す.
        self._pred_hist[self.T-1] = pred
        #for t in range(T):
            #pred = self._predict_with_formula(data_x,data_y)  # maxlag=0の場合1行目を取り出す.
            #self._pred_hist[t] = pred
            #self._pred_hist_formulas[t] = preds_f


    def _update_score(self, label, eval_lookback: int, eval_method="default") -> None:
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
            self.score = np.sum(np.nan_to_num( (np.sign(self._pred_hist)[-eval_lookback:] * label[-eval_lookback:]) ))
        elif eval_method == "acc":
            self.score = np.sum(np.nan_to_num((np.sign(self._pred_hist)[-eval_lookback:] * np.sign(label[-eval_lookback:]))))
#sign 大于0等于1 小于0等于-1 等于0等于0
        else:
            raise NotImplementedError(f"unknown eval_method : {eval_method}")


    def recalc(self, data_x,data_y, label, eval_lookback, eval_method="default"):
        """ 予測値の履歴をリセット計算し直す. 合わせて評価値も計算し直す.
        Args:
            feature_arr_block (np.ndarray):
            return_arr (np.ndarrayeval_method, optional): [description]. Defaults to "default".
        Returns:
            Trader: self
        """
        #self._recalc_predicts_hist(data_x,data_y)
        self._update_score(label, eval_lookback, eval_method)
        return self


    def _append_predicts_hist(self, data_x,data_y) -> None:
        """ トレーダおよび保持しているFormulaの予測値履歴の末尾を追記.
        跟踪器和持有的Formula预测值历史的末尾。
        Args:
            feature_arr (np.ndarray):
        Effects:
            self._pred_hist
            self._pred_hist_formulas
        """
        #self._time_index += 1
        pred = self.predict(data_x,data_y)
        self._pred_hist[self.T] = pred
        self.T = self.T + 1
        #self._pred_hist_formulas = np.append(self._pred_hist_formulas, preds_f, axis=0)


    def _update_weights(self, x_train,y_train,x_valid, y_valid,evals_result=dict(),save_path=None,) -> None:

        # train

        self.Model.logger.info("start...")
        self.Model.fitted = True
        #for step in range(self.Model.n_epochs):
            #self.Model.logger.info("Epoch%d:", step)
        self.Model.logger.info("training...")
        self.Model.train_epoch(x_train, y_train)
        self.Model.logger.info("evaluating...")
        self.train_loss, self.train_score = self.Model.test_epoch(x_train, y_train)
        self.val_loss, self.val_score = self.Model.test_epoch(x_valid, y_valid)
        self.Model.logger.info("train %.6f, valid %.6f" % (self.train_score, self.val_score))
        self.score=self.val_score
        evals_result["train"].append(self.train_score)
        evals_result["valid"].append(self.val_score)

        if self.val_score > self.best_score:
            self.best_score = self.val_score
            self.stop_steps = 0
            #best_epoch = step
            #self.best_param = copy.deepcopy(self.Model.model.state_dict(),self.Model.model2.state_dict())
            self.best_param = self.Model
            self.stop_steps += 1
            if self.stop_steps >= self.Model.early_stop:
                self.Model.logger.info("early stop")
#########

        # with R.start(experiment_name=EXP_NAME):
        #     self.Model.fit(dataset)
        #     R.save_objects(trained_model = self.Model)
        #
        #     rec = R.get_recorder()
        #     rid = rec.id  # save the record id
        #
        #     # Inference and saving signal
        #     sr = SignalRecord(self.Model, dataset, rec)
        #     sr.generate()


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