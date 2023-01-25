# Basit Lineer Regresyon
# Kütüphanelerin eklenmesi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Veri setinin okunarak bağımlı ve bağımsız değişkenlere ayrılması
dataset = pd.read_csv('MaasVerisi.csv', encoding = 'iso-8859-9')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
y = y.reshape(-1, 1) #bir boyutlu array olması ölçekleme sırasında problem oluşturduğundan bu değişiklik yapıldı

# Veri setinin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

 
# Eğitim verileri ile modelin eğitilmesi
#basit leneer regresyon kullanıyoruz burada tek bir kolondan tek bir kolon tahmin etmeye çalışıyoruz çünkü  birdenf azla kolondan tahmin etmeye çalışsaydık
#multiple lineer regresyon kullanırdık
from sklearn.linear_model import LinearRegression #bu kütüphane basiti de multiple regresyonunu da yapıyor
#verinin tipine göre karar veriyor kütüphane basit mi multiple mı kullanacağına
regressor = LinearRegression()
regressor.fit(X_train, y_train) #eğitimi fit ettik uyguladık yani

# Test verileri kullanılarak tahmin yapılması
y_pred = regressor.predict(X_test)
#fit ile eğittik predict ile tahmin ettiriyoruz
#test kısmıyla tahmin edecek

#Test değerleri ile tahmin değerleri arasındaki hata farkının hesaplanması
#Modelin başarısının ölçüldüğü kısım.
from sklearn.metrics import r2_score
Rkare= r2_score(y_test, y_pred)
#yani bi nevi başarısını ölçüyoruz algoritmanın
#1'e yakın olması çok iyi demek

# Eğitim sonuçlarının görselleştirilmesi (sınavda çıkmayacak)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Maas vs Tecrube (Eğitim Bolumu)')
plt.xlabel('Yillik Tecrube')
plt.ylabel('Maas')
plt.show()

# Test sonuçlarının görselleştirilmesi (sınavda çıkmayacak)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Maas vs Tecrube (Test Bolumu)')
plt.xlabel('Yillik Tecrube')
plt.ylabel('Maas')
plt.show()
