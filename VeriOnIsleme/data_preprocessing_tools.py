# Veri Ön İşleme

# Kütüphane ekleme
import numpy as np #matematiksel işlemleri yapar
import matplotlib.pyplot as plt #çizimle ilgili bir kütüphane, verilerimizi grafiğe falan dökmeye yara
import pandas as pd #scikitLearn altında yer alır veri işlemleri için kullanılır veri okuma,data frame'e dönüştürme vs


# Veri setimizi okuyoruz
dataset = pd.read_csv('Data.csv') #verisetinin adını belirttik ve okuyoruz, pandası veri işlemleri için kullanırız demiştik
#yukarıdaki kodla verileri data frame şekline çevirdik(çerçevelettik yani)
X = dataset.iloc[:, :-1].values #bağımsız değişken yani var olan bilgilerdir
y = dataset.iloc[:, -1].values# bağımlı değişken yani var olan bilgilerle tahmin edeceğimiz bilgi
# iloc veri setinin belirli parçalarını okumaya yarar
# iloc[okuyacağımız satırlar:okuyacağımız sütunlar], -1 son kolonu ifade eder ,:-1 demek son kolon dahil değil baştan hepsini al demek
# ,-1 demek te direk son kolonu al demek
#verilen verideki ilk sütun sıfırıncı indekse sahip oluyor
print(X)
print(y)

# 1.adım eksik verileri tamamlamak
from sklearn.impute import SimpleImputer #bu kütüphande boş olan değerleri hangi strateiyle doldurmak istersek otomatik yapıyor onu
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# SimpleImputer'da birçok parametre vardır bizim 2 tane kullanmamız demek diğer parametrelerin default olarak kaydedilmesi demektir.
# (missing değerler nan olarak olanlardır,mean yani ortalama stratejisiyle doldur ) kullandığımız parametrelerin anlamları
# ortalama stratejisi o kolonun ortalamasını al boş yerlere onları geçir demek
imputer.fit(X[:, 1:3]) # fit etmek demek yaptığımız şeyi uygula demek
#X'in tüm satırlarına bak ve 1 ve 2.kolonlardaki boşlukları doldur
X[:, 1:3] = imputer.transform(X[:, 1:3]) # X'teki eski tabloyu yaptığımız tabloyla transforme t yani atama yapıyor işte
print(X)

#2.adım kategorik verileri sayısal forma çevirmek
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder #onehatencoder yöntemini kullanacağız
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') #[0] ile sıfırıncı kolona uyguluyor işlemi
X = np.array(ct.fit_transform(X))
print(X) #ülkeler 1 0 0 diye sayısal formatlara çevrildi o ülkenin satırıysa 1 diğer kolonlar sıfır yani kaç ülke varsa o kadar yeni kolon geldi başa
# Y kolonundaki yes, no'ları sayısala çeviriyoruz burda farklı bir şey kullanıyoruz çünkü sadece 2 değişken var
from sklearn.preprocessing import LabelEncoder # yesleri 1, no'ları sıfır yaptı labelencoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# verileri train ve test olarak bölüyoruz yani veri setini 4 e böldük
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) # text_size derken kaçta kaçı test olsun onu belirttik, train de buna otomatik olarak hesaplanıyor
print(X_train)
print(X_test)
print(y_train)
print(y_test)
#eğitim test mantığı şöyle yani bizim 10 verimiz varsa algoritmaya bunun 8 tanesini öğretiriz eğitiriz
#sonra algoritmadan bize kalan 2 değeri tahmin etmesini isteris test ederiz, 2 değer zaten bizde vardı
#onları algoritmanın tahmin ettiği değerle karşılaştırıp algoritmanın başarısını test etmiş oluruz

# 3.adım veri ön işeleöe son adım > verilerin arasındaki sayılar uçurumdaysa onları belirli bir aralığa sıkıştırıyoruz
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
# y değerleri 0 ve 1 olduğu için scaler yapmaya gerek duymadık, x'leri -1 ve 1 arasında yerleştirdik
