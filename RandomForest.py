# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Ensemble = Farklı algroitmayı yada aynı algoritmayı tekrar tekrar gerçekleştirirek yapılan model
# Yani buna Ensembling Regression Model denir.

data = pd.read_csv("positions.csv")

level = data.iloc[:,1].values.reshape(-1,1) # çift boyutlu old. için reshape(-1,1) kullandık.
salary = data.iloc[:,2].values # y ekseni tek boytulu olması gerektiği için reshape kullanmadık.

#burada RandomForestRegressor classını yukarda tanımladığımız için çağırıyoruz.
#n_estimators = Bu kısım kaç tane decision tree(Karar Ağacı) oluşturmamıza yardımcı oluyor.
#n_estimators ın değerini arttırırsanız random ve hata oranı azalıyor. Gerçekçi bir hal alıyor.
#n_estimators=5,random_state=0 : burda random_ state ile de random durumunu sabitliyoruz ve tek bir sonuçta kalmasını sağlıyor.
regression = RandomForestRegressor(n_estimators=5,random_state=0)
regression.fit(level,salary)

print(regression.predict([[8.3]]))



