# -*- coding: utf-8 -*-

# Bu bölümde csv dosyasını yüklemek için pandası kullandık.
import pandas as pd
#Grafikte görselleştirme yapmak için matplotlib.pyplot kütüphanesini kullanıyoruz
import matplotlib.pyplot as plt
dataFrame = pd.read_csv("hw_25000.csv")

print(dataFrame.columns)
print(dataFrame.head(10))
# plt.scatter ile veriyi görselleştirdik.
plt.scatter(dataFrame.Height,dataFrame.Weight)
# X ve Y eksenlerini diyagramın formatına göre isimlendirdik. 
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.show()

