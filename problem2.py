import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.cm as cm
import os

clf = linear_model.LinearRegression()

cherry_data = pd.read_csv('tokyo.csv')
bloom_data = pd.read_csv('blooming.csv')

"""
test = cherry_data.query("year == 1966|year == 1971|year == 1985|year == 1994|year == 2008")
train = cherry_data.loc[set(cherry_data.index) - set(test.index), :]
ans_test = bloom_data.query("year == 1966|year == 1971|year == 1985|year == 1994|year == 2008")
ans_train = bloom_data.loc[set(bloom_data.index) - set(ans_test.index), :]
"""

s_year = 1961
e_year = 2017
y_list = []
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
    y_list.append(year)
    s_list.append(start)
    d_list.append(march_d)
    Dj_list.append(int(Dj))

if os.path.isfile("output2.png") is False:
    plt.plot(y_list, Dj_list)
    plt.title("the counting day of the year")
    plt.xlabel("year")
    plt.ylabel("counting from 01/01")
    plt.plot([s_year, e_year], [32, 32], linestyle="--", color="#FF0000")
    filename = "output2.png"
    plt.savefig(filename)
    plt.show()

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


if os.path.isfile("output3.png") is False:
    plt.plot(np.arange(5, 40), data, color="#AAAAAA")
    plt.plot(np.arange(5, 40), mean_DTS, color="#FF0000")
    plt.title(str(s_year) + " to " + str(e_year))
    plt.xlabel("Ea")
    plt.ylabel("DTS")
    filename = "output3.png"
    plt.savefig(filename)
    plt.show()

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

#後ろ(03/31)から数えて
ave_bloom = int(np.array(d_list).mean() - np.array(ans_list_train).mean() + 0.5)
other_data_train = pd.DataFrame()
other_data_test = pd.DataFrame()
for year in y_list:
    Djs = np.array(s_list) + np.array(Dj_list)
    Djs_end = np.array(s_list) + np.array(d_list)
    if year == 1966 or year == 1971 or year == 1985 or year == 1994 or year == 2008:
        other_data_test = other_data_test.append(cherry_data.ix[Djs[year - 1961]:Djs_end[year - 1961] - ave_bloom, \
                                                 ["最高気温", "現地平均気圧"]].sum(), ignore_index = True)
    else:
        other_data_train = other_data_train.append(cherry_data.ix[Djs[year - 1961]:Djs_end[year - 1961] - ave_bloom, \
                                                   ["最高気温", "現地平均気圧"]].sum(), ignore_index=True)

clf.fit(pd.concat([pd.DataFrame(data[opt_Ea - 5, :]), other_data_train], axis=1), \
        pd.DataFrame(ans_list_train))
print("coef ", clf.coef_)
print("intercept ", clf.intercept_)
print("score", clf.score(pd.concat([pd.DataFrame(DTSj_list_test), other_data_test], axis=1), \
                         pd.DataFrame(ans_list_test)))