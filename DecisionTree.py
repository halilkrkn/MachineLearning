# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("positions.csv")

#x ekseni içim
level = data.iloc[:,1].values.reshape(-1,1)
#y ekseni için
salary =data.iloc[:,2].values.reshape(-1,1)

regression = DecisionTreeRegressor()

regression.fit(level,salary)

#burada 8.50 ve küçük değerler level 8 in salary degerini gösteriyor.
print(regression.predict([[8.50]]))
# 8.51 den büyük değer verirsek eğer (en son 8.99) bu da level 9 un salary karşılığını verdi.
print(regression.predict([[8.51]]))

# Grafiğimizin x ekseni olarak level y ekseni olarak salary ve color olarak mor seçip görselleştirdik.
plt.scatter(level,salary, color ="purple")


x = np.arange(min(level),max(level),0.01).reshape(-1,1)
plt.plot(x,regression.predict(x),color="orange")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Decision Tree Model")
plt.show()