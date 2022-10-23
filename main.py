import numpy as np


def get_peaks_troughs(h, rangesize):
    peaks = list()
    troughs = list()
    S = 0
    for x in range(1, len(h) - 1):
        if S == 0:
            if h[x] > h[x + 1]:
                S = 1  ## down
            else:
                S = 2  ## up
        elif S == 1:
            if h[x] < h[x + 1]:
                S = 2
                ## from down to up
                if len(troughs):
                    ## check if need merge
                    (prev_x, prev_trough) = troughs[-1]
                    if x - prev_x < rangesize:
                        if prev_trough > h[x]:
                            troughs[-1] = (x, h[x])
                    else:
                        troughs.append((x, h[x]))
                else:
                    troughs.append((x, h[x]))


        elif S == 2:
            if h[x] > h[x + 1]:
                S = 1
                ## from up to down
                if len(peaks):
                    prev_x, prev_peak = peaks[-1]
                    if x - prev_x < rangesize:
                        if prev_peak < h[x]:
                            peaks[-1] = (x, h[x])
                    else:
                        peaks.append((x, h[x]))
                else:
                    peaks.append((x, h[x]))

    return peaks, troughs


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import baostock as bs

    lg = bs.login()
    filename = "sh.601699"
    rs = bs.query_history_k_data_plus(filename,
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                      start_date='2021-01-01',
                                      end_date='2022-01-01',
                                      frequency="d", adjustflag="3")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()
    h=result['close'].values.astype(float)
    peaks, troughs = get_peaks_troughs(h, 10)
    plt.plot(np.arange(len(h)), h)
    for x, y in peaks:
        plt.text(x, y, y, fontsize=10, verticalalignment="bottom", horizontalalignment="center")
    for x, y in troughs:
        plt.text(x, y, y, fontsize=10, verticalalignment="top", horizontalalignment="center")
    plt.show()