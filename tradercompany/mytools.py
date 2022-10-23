class mytools:
    def make_peak(start_time="2010-01-01",val_time="2021-01-02",train_end="2021-01-01",end_time="2021-06-01",stock="sh600519",filename="test.pkl"):
        import qlib
        qlib.init()
        infer_processors=[{
            "class": "FilterCol",
            "kwargs": {"fields_group": "feature",
                      "col_list": ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
                                    "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
                                    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
                                ]}},{
            "class": "RobustZScoreNorm",
            "kwargs": {"fields_group": "feature",
                      "clip_outlier": "true"

            }},{"class": "Fillna",
                  "kwargs":{
                      "fields_group": "feature"
                  }
                },

        ]
        learn_processors=[{'class': 'DropnaLabel'},{'class': 'CSRankNorm','kwargs':{
                      'fields_group': 'label'}}]
        handler_kwargs = {
                "start_time": start_time,
                "end_time": end_time,
                "fit_start_time": start_time,
                "fit_end_time": end_time,
                "instruments":  [stock],#MARKET,
                "infer_processors":infer_processors,
                "learn_processors":learn_processors
        }
        handler_conf = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": handler_kwargs,

        }
        from qlib.utils import init_instance_by_config

        hd = init_instance_by_config(handler_conf)
        import numpy as np


        def get_peaks_troughs(h, rangesize):
            peaks = list()
            troughs = list()
            S = 1
            for x in range(1, len(h) - 5):
                if S == 0:
                    if h[x] > 0.2*(h[x + 1]+h[x + 2]+h[x + 3]+h[x + 4]+h[x + 5]):
                        S = 1  ## down
                    else:
                        S = 2  ## up
                elif S == 1:
                    if h[x] < 0.2*(h[x + 1]+h[x + 2]+h[x + 3]+h[x + 4]+h[x + 5]):
                        S = 2
                        ## from down to up
                        if len(troughs):
                            ## check if need merge
                            (prev_x, prev_trough) = troughs[-1]
                            if x - prev_x < rangesize:
                                if prev_trough > h[x]:
                                    troughs[-1] = (x, h[x])
                            else:
                                #if(len(peaks) and h[x]<peaks[-1]):
                                troughs.append((x, h[x]))
                        else:

                            troughs.append((x, h[x]))


                elif S == 2:
                    if h[x] > 0.2*(h[x + 1]+h[x + 2]+h[x + 3]+h[x + 4]+h[x + 5]):
                        S = 1
                        ## from up to down
                        if len(peaks):
                            prev_x, prev_peak = peaks[-1]
                            if x - prev_x < rangesize:
                                if prev_peak < h[x]:
                                    peaks[-1] = (x, h[x])
                            else:
                                #if(len(troughs) and h[x]>troughs[-1]):
                                peaks.append((x, h[x]))
                        else:
                            peaks.append((x, h[x]))

            return peaks, troughs
        import numpy as np
        from matplotlib import pyplot as plt
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import baostock as bs

        lg = bs.login()
        rs = bs.query_history_k_data_plus(stock[:2]+'.'+stock[2:],
                                          "date,close",
                                          start_date=start_time,
                                          end_date=end_time,
                                          frequency="d", adjustflag="3")
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        bs.logout()
        h=result['close'].values.astype(float)
        peaks, troughs = get_peaks_troughs(h, 10)


        result.loc[:, 'label'] = 0
        result['label'].astype(float)
        for x, y in peaks:
            result['label'][x] = -1
        for x, y in troughs:
            result['label'][x] = 1
        s = 1
        for i in range(len(result['label'].values)):
            if result['label'].values[i] == 0:
                pass
            if result['label'].values[i] == 1 and s == 0:
                result['label'].values[i] = 0
            elif result['label'].values[i] == -1 and s == 1:
                result['label'].values[i] = 0
            elif result['label'].values[i] == -1 and s == 0:
                s = 1
            elif result['label'].values[i] == 1 and s == 1:
                s = 0


        dataset_conf = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": hd,
                "segments": {
                    "train": (start_time, train_end),
                    "valid": (val_time, end_time),
                    "test": ("2022-12-20", "2022-12-31"),
                },
            }
        }
        dataset = init_instance_by_config(dataset_conf)

        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            #data_key=DataHandlerLP.DK_L,
        )
        df_train['label'] = result['label'].values[0:len(df_train['label'])]
        df_valid['label'] = result['label'].values[len(df_train['label']):]
        df=[df_train,df_valid]
        #df = df_train.append(df_valid)
        #print(df)
        import pickle
        a_file = open(filename, "wb")
        pickle.dump(df, a_file)
        a_file.close()
    def read_pkl(filename='test.pkl'):
        import pickle
        a_file = open(filename, "rb")
        df = pickle.load(a_file)
        a_file.close()
        return df
