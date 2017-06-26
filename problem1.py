import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

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
t_list = []
t_list_test = []
d_list = []
d_list_test = []
for year in range(s_year, e_year + 1):
    if year == 1966 or year == 1971 or year == 1985 or year == 1994 or year == 2008:
        start = test.query("year==" + str(year) + "&month==" + str(2) + "&day==" + str(1)).index[0]
        end = test.query("year==" + str(year) + "&month==" + str(ans_test[ans_test.year == year].month.values[0]) + \
                          "&day==" + str(ans_test[ans_test.year == year].day.values[0])).index[0]
        t_list_test.append(train.ix[start:end, "最高気温"].sum())
        d_list_test.append(int(end - start))
        continue
    start = train.query("year=="+str(year)+"&month=="+str(2)+"&day=="+str(1)).index[0]
    end = train.query("year=="+str(year)+"&month=="+str(ans_train[ans_train.year == year].month.values[0])+\
                      "&day=="+str(ans_train[ans_train.year == year].day.values[0])).index[0]
    y_list.append(year)
    t_list.append(train.ix[start:end, "最高気温"].sum())
    d_list.append(int(end - start))

mean = np.mean(t_list)
plt.plot(y_list, t_list)
plt.title("the sum of max temp until blooming (mean: "+ str(mean) + ")")
plt.xlabel("year")
plt.ylabel("the sum of temp")
plt.plot([s_year, e_year], [600, 600], linestyle="--", color="#dddddd")
filename = "output.png"
plt.savefig(filename)


clf.fit(pd.DataFrame([x - 600 if x > 600 else 0 for x in t_list]), pd.DataFrame(d_list))
print("mean: 600", clf.score(pd.DataFrame([x - 600 if x > 600 else 0 for x in t_list_test]), pd.DataFrame(d_list_test)))
clf.fit(pd.DataFrame([x - mean if x > mean else 0 for x in t_list]), pd.DataFrame(d_list))
print("mean: " + str(int(mean)), clf.score(pd.DataFrame([x - mean if x > mean else 0 for x in t_list_test]), pd.DataFrame(d_list_test)))
