import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.cm as cm
import os

clf = linear_model.LinearRegression()

cherry_data = pd.read_csv('tokyo.csv')
bloom_data = pd.read_csv('blooming.csv')

test = cherry_data.query("year == 1966|year == 1971|year == 1985|year == 1994|year == 2008")
train = cherry_data.loc[set(cherry_data.index) - set(test.index), :]
ans_test = bloom_data.query("year == 1966|year == 1971|year == 1985|year == 1994|year == 2008")
ans_train = bloom_data.loc[set(bloom_data.index) - set(ans_test.index), :]

s_year = 1961
e_year = 2017
y_list = []
y_list_train = []
s_list = []
d_list = []
Dj_list = []
ans_list_train = []
ans_list_test = []
for year in range(s_year, e_year + 1):
    start = cherry_data.query("year=="+str(year)+"&month=="+str(1)+"&day=="+str(1)).index[0]
    march_end = cherry_data.query("year=="+str(year)+"&month==" + str(3) + "&day==" + str(31)).index[0]
    end = cherry_data.query("year==" + str(year) + "&month==" + str(bloom_data[bloom_data.year == year].month.values[0]) + \
                      "&day==" + str(bloom_data[bloom_data.year == year].day.values[0])).index[0]
    ave_t = np.mean(cherry_data.ix[start:march_end, "平均気温"])
    N = 35 + 40 / 60
    Dj = 136.75 - 7.689 * N + 0.133 * (N ** 2) - 1.307 * np.log(4) + \
         0.144 * ave_t + 0.285 * (ave_t ** 2)
    march_d = int(march_end - start)
    if year == 1966 or year == 1971 or year == 1985 or year == 1994 or year == 2008:
        ans_list_test.append(int(end - start))
    else:
        ans_list_train.append(int(end - start))
        y_list_train.append(year)
    y_list.append(year)
    s_list.append(start)
    d_list.append(march_d)
    Dj_list.append(int(Dj))

Ts = 17 + 273.15
R = 8.314
DTSj_list = []
for year in y_list:
    if year == 1966 or year == 1971 or year == 1985 or year == 1994 or year == 2008:
        continue
    for Ea in range(5, 40):
        DTS = 0
        for i in range(Dj_list[year - s_year], d_list[year - s_year]):
            Tij = cherry_data.ix[s_list[year - s_year] + i, "平均気温"] + 273.15
            DTS = DTS + np.exp((Ea * (Tij - Ts)) / (R * Tij * Ts))
        DTSj_list.append(DTS)

data = np.array(DTSj_list)
data = data.reshape(-1, 35).T
mean_DTS = []
for Ea in range(35):
    mean_DTS.append(np.average(data[Ea, :]))


min = 100
opt_Ea = 0
for Ea in range(35):
    rmse = np.sqrt(np.sum((data[Ea, :] - mean_DTS[Ea]) ** 2) / len(data[Ea, :]))
    if rmse < min:
        min = rmse
        opt_Ea = 5 + Ea

print("optimal Ea", opt_Ea)

DTSj_list_test = []
for year in y_list:
    if year == 1966 or year == 1971 or year == 1985 or year == 1994 or year == 2008:
        DTS = 0
        for i in range(Dj_list[year - s_year], d_list[year - s_year]):
            Tij = cherry_data.ix[s_list[year - s_year] + i, "平均気温"] + 273.15
            DTS = DTS + np.exp((opt_Ea * (Tij - Ts)) / (R * Tij * Ts))
        DTSj_list_test.append(int(DTS))


clf.fit(pd.DataFrame(data[opt_Ea - 5, :]), pd.DataFrame(ans_list_train))
print("coef ", clf.coef_)
print("intercept ", clf.intercept_)
print("score", clf.score(pd.DataFrame(DTSj_list_test), pd.DataFrame(ans_list_test)))

#休眠打破後、400超えるまでの値と日照時間とか
#後ろ(03/31)から数えて
ave_bloom = int(np.array(d_list).mean() - np.array(ans_list_train).mean() + 0.5)
y_sum_data = pd.DataFrame()
for year in y_list:
    Djs = np.array(s_list) + np.array(Dj_list)
    Djs_end = np.array(s_list) + np.array(d_list)
    if year == 1966 or year == 1971 or year == 1985 or year == 1994 or year == 2008:
        continue
    else:
        y_sum_data = y_sum_data.append(cherry_data.ix[Djs[year - 1961]:Djs_end[year - 1961] - ave_bloom, :].sum(), ignore_index=True)

for x in ["現地平均気圧", "海面平均気圧"]:
    plt.plot(y_list_train, y_sum_data[x] / 1000)
    plt.plot(y_list_train, ans_list_train, color="#000000")
    filename = x + ".png"
    plt.savefig(filename)
    plt.show()

for x in ["合計降水量", "1時間最大降水量", "10分間最大降水量", "最低気温", "平均湿度", "最低湿度"]:
    plt.plot(y_list_train, y_sum_data[x] / 1000)
    plt.plot(y_list_train, y_sum_data[x] / 100)
    plt.plot(y_list_train, y_sum_data[x] / 10)
    plt.plot(y_list_train, ans_list_train, color="#000000")
    filename = x + ".png"
    plt.savefig(filename)
    plt.show()
