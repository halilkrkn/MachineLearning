# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures 
data = pd.read_csv("positions.csv")

print(data.columns)
print("************\n")
#print(data.describe())

#X ekseni için
Level = data.iloc[:,1].values.reshape(-1,1)
#Y ekseni için
Salary =data.iloc[:,2].values.reshape(-1,1) 

#BU kısımda Linear bir doğru üzerinden tahmin yaptık ve linear bir line çizdik.
regression = LR()
#X ve Y eksini fitledik.
regression.fit(Level,Salary)

# BUrada Level i 8.3 olan bir kişi için tahmini Salary fiyatını Hesapladık.
tahmin = regression.predict([[8.3]])

#burada linear bir doğrusallık yaptığımız için tahmin sağlıklı bir tahmin değil.
print("Tahmini Salary fiyatı:"+str(tahmin))

# Bu Polinomal bir eğri ile tahmin yapıtık.

regressionPoly = PolynomialFeatures(degree = 4)
# burada Level değeri üzerinde polynomal bir eğri oluşturduk.
levelPoly = regressionPoly.fit_transform(Level)

regression2 = LR()
regression2.fit(levelPoly,Salary)

# burda daha yakın tahminler yapmak için polynomal bir eğri ile sonuca yaklaştık
tahmin2 = regression2.predict(regressionPoly.fit_transform([[8.3]]))
print("\nTahmin2 nin Salary fiyatı:"+str(tahmin2))

## BU kısımlarda hem linear hemde polinomal bir eğri ile nasıl bir tahmin yaklaşımını yaptığımızı gördük.
plt.scatter(Level,Salary, color="orange")
plt.plot(Level,regression.predict(Level),color="purple")
plt.plot(Level,regression2.predict(levelPoly))
plt.show()




