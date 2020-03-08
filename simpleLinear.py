# -*- coding: utf-8 -*-

# Bu bölümde csv dosyasını yüklemek için pandası kullandık.
import pandas as pd
import numpy as np
#Grafikte görselleştirme yapmak için matplotlib.pyplot kütüphanesini kullanıyoruz
import matplotlib.pyplot as plt
# sklearn.linear_model  kütüphanesinden LinearRegression Modülünü (class gibi) çağırdık.
from sklearn.linear_model import LinearRegression
# Yüzdelik Doğruluğu Hesaplamak için kullanıyoruz.
from sklearn.metrics import r2_score
dataFrame = pd.read_csv("hw_25000.csv")

boy = dataFrame.Height.values.reshape(-1,1)
kilo = dataFrame.Weight.values.reshape(-1,1)
# burada bir instance oluşturduk. bir nevi Yukarıda tanımladığımız LinearRegression classını çağırdık.
regression = LinearRegression()
# boy ve kiloya göre line fit elde etmek için fit() fonk. kullandık.
regression.fit(boy,kilo)


print("60 Kilo:"+str(regression.predict([[60]])))
print("62 Kilo:"+str(regression.predict([[62]])))
print("64 Kilo:"+str(regression.predict([[64]])))
print("66 Kilo:"+str(regression.predict([[66]])))
print("68 Kilo:"+str(regression.predict([[68]])))
print("70 Kilo:"+str(regression.predict([[70]])))
# Yukardaki kodda fit line üzerinden tahmin yaparak line yani kiloyu buluyoruz.
#Çıktı:
#60 Kilo:[[102.43284366]]
#62 Kilo:[[108.59979655]]
#64 Kilo:[[114.76674944]]
#66 Kilo:[[120.93370233]]
#68 Kilo:[[127.10065522]]
#70 Kilo:[[133.26760811]]


print("-***********************************")
print(dataFrame.columns)
print(dataFrame.head(10))
# plt.scatter ile veriyi görselleştirdik.
plt.scatter(dataFrame.Height,dataFrame.Weight,color="orange")

# bu kısımda Kilo ekseninin min ve max değerlerini baz alarak tahmini Line ı çizdik.
x = np.arange(min(dataFrame.Height), max(dataFrame.Height)).reshape(-1,1)
# Yani burda her X için Y e eşit olacak şekilde tahmini line(çizgi) nı çizmesini sağladık.
plt.plot(x,regression.predict(x),color=("purple"))

# X ve Y eksenlerini diyagramın formatına göre isimlendirdik. 
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Simple Linear Regression Models")
plt.show()
print("-***********************************")
#Dogruluk payını hesapladık.
print("Yüzde Doğruluk:"+str(r2_score(kilo,regression.predict(boy))))
#Çıktı:
#Yüzde Doğruluk:0.2528666917428809 yani elimizdeki data %25 doğruluk payına sahip bir veri.







