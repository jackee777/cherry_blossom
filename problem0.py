from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

base_url = 'http://www.data.jma.go.jp/sakura/data/'
next_url = 'sakura003_06.html'
data = pd.DataFrame()
while True:
    htmlData = urlopen(base_url + next_url).read()
    htmlData = htmlData.decode('utf-8', 'ignore')
    htmlParsed = BeautifulSoup(htmlData, 'html.parser')
    obj = htmlParsed.find_all("div", class_="indent")

    next_url = obj[0].find("a").get("href")
    inner = obj[0].find("pre")
    table = inner.text.split("\n")
    for i in table:
        row = re.split(" +", i)
        if "地点名" in row[0]:
            year  = []
            for j in range(1, len(row) - 2):
                year.append(row[j])
        if "東京" in row[0]:
            month  = []
            day = []
            for j in range(2, len(row) - 3):
                if j % 2 == 0:
                    month.append(row[j])
                if j % 2 == 1:
                    day.append(row[j])
    data = data.append(pd.DataFrame([year, month, day]).T)
    time.sleep(1)
    if next_url == "sakura003_00.html":
        break

data = data.append(pd.DataFrame(["2017","3", "21"]).T)
data.columns = ["year", "month", "day"]
data = data.sort_values(by="year")
data = data.reset_index(drop=True)
data.to_csv("blooming.csv")