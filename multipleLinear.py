# -*- coding: utf-8 -*-

# Bu bölümde csv dosyasını yüklemek için pandası kullandık.
import pandas as pd
import numpy as np
#Grafikte görselleştirme yapmak için matplotlib.pyplot kütüphanesini kullanıyoruz
import matplotlib.pyplot as plt
# sklearn.linear_model  kütüphanesinden LinearRegression Modülünü (class gibi) çağırdık.
from sklearn.linear_model import LinearRegression

dataFrame = pd.read_csv("insurance.csv")

print(dataFrame.columns)

# Datamız hakkında bilgi almak için kullandık.
#describe = dataFrame.describe()
#print(describe)

# verimizdeki expenses kısmının değerlerini çektik.
# y ekseni için bu kısmı oluşturduk.
expenses = dataFrame.expenses.values.reshape(-1,1)

# bu kısımda iloc ile istediğimiz sütunları seçtik ve o dataları sıraladık.
# X ekseni 
ageBmis = dataFrame.iloc[:,[0,2]].values

regression = LinearRegression()

regression.fit(ageBmis,expenses)

## Age ve Bmis kişilerinini ortalama harcamaları bu düzeyde.
print(regression.predict([[20,20]]))

## Bmis(Vücut kitle indexi) arttıkça harcamalarda nasıl bir değişiklik oluyor onu gözlemledik.
print(regression.predict([[20,20],[20,21],[20,22],[20,23],[20,24],[20,25]]))

























