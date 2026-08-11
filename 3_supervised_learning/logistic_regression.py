"""
Amaç:
    -UCI Heart Disease veri setini kullanarak lojistik regresyon modeli ile ikili sınıflandırma problemi çözme
    -Model, bir bireyin kalp hastlaığına sahip olup olmadığını tahmin etmeyi amaçlar ve accuracy metriği ile değerlendirir

Veri Seti:
    -UCI ML Repo:"https://archive.ics.uci.edu/dataset/45/heart+disease"
    -Veri Seti bireylere ait demografik ve klinik ölçümlerini içeriyor.
    -Features: Yaş, Cinsiyet,Ağrı Tipi,Kolestrol,Kan Basıncı vb.

Plan/Program:
    -Veri seti yükle temel analizleri yap
    -Veri seti içerisinde eksik değer kontrolü yap gerekirse temizle
    -Öznitelik ve hedef değişkenlerin ayrılması
    -Eğitim ve test veri setlerinin oluşturulması
    -Lojistik regresyon modelinin tanımlanması ve eğitilmesi
    -Modelin Test veri seti ile değerlendirilmesi

Kurulum:
    pip install pandas scikit-learn ucimlrepo
"""
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Veri seti yükle temel analizleri yap

heart_disease = fetch_ucirepo(id=45) 
df = pd.DataFrame(data = heart_disease.data.features)
df["target"] = heart_disease.data.targets
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)
print(df.head())

#Veri seti içerisinde eksik değer kontrolü yap gerekirse temizle

if df.isnull().any().any():
    df.dropna(inplace=True)
    eksik_sayisi = df.isnull().sum().sum()
    print(f"{eksik_sayisi} adet NaN değer bulundu.")
    print("NaN değerler veri setinden çıkartıldı.")
else:
    print("NaN değer bulunmuyor.")

#Öznitelik ve hedef değişkenlerin ayrılması
X = df.drop(["target"], axis = 1).values #features
y = df.target.values

#Eğitim ve test veri setlerinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

#Lojistik regresyon modelinin tanımlanması ve eğitilmesi
log_reg = LogisticRegression(penalty="l2", C =1, max_iter=100)
log_reg.fit(X_train, y_train)

#Modelin Test veri seti ile değerlendirilmesi
acc = log_reg.score(X_test, y_test)
print(f"Accuracy: {acc}")