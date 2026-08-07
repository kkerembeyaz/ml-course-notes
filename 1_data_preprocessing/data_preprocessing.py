"""
Makine Öğrenmesi Veri Ön İşleme Pratikleri
Amaç:
    1.Eksik veri tespiti, Çıkartılması ve uygun değerler ile doldurma
    2.IQR Yöntemiyle sayısal sütunlardaki aykırı değerleri tespit etmek
    3.Kategorik verileri label encoding ve one-hot encoding ile dönüştür.
    4.Veriyi Train-Test-Validation ayir.
    5.Sayısal özelliklere standartization ve normalization uygula

Kurulum
pip install pandas scikit-learn
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, minmax_scale, MinMaxScaler

df = pd.read_csv("musteri_verisi_ml_pratik.csv")

print(df.head())
print(df.info())

#3. Eksik Veri Analizi

print(df.isnull().sum())

df_dropna = df.dropna() #Eksik veri çıkart
print(f"Eksik veriler çıktıktan sonra: \n{df_dropna}")


df_filled = df.copy()
sayisal_Sutunlar = ["yas", "maas", "deneyim_yili"]

#Sayısal sutunları medyan ile doldurma
for sutun in sayisal_Sutunlar:
    medyan_Degeri = df_filled[sutun].median()
    df_filled[sutun] = df_filled[sutun].fillna(medyan_Degeri)

#Kategorik sutunları en sık tekrar eden deger ile doldur

df_filled["egitim"] = df_filled["egitim"].fillna(df_filled["egitim"].mode()[0])
print(f"Eksik veriler doldurulduktan sonra:\n {df_filled}")

#4.IQR ile Aykırı değerleri tespit etme

aykiri_deger_maskesi = pd.Series(False, index = df_filled.index)

for sutun in sayisal_Sutunlar:

    q1 = df_filled[sutun].quantile(0.25)
    q3 = df_filled[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df_filled[sutun] < alt_sinir) | (df_filled[sutun] > ust_sinir)
    )

    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi

    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")

    if sutun_maskesi.any():
        print(f"Aykırı değerler: \n{df_filled.loc[sutun_maskesi, sutun]}")

print(f"En az bir aykırı değer içeren satırlar \n{df_filled.loc[aykiri_deger_maskesi]}")

# Aykırı Değer İçeren Satırları Veri Setinden Çıkarma

df_clean = df_filled.loc[~aykiri_deger_maskesi].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"Aykırı değerler çıktıktan sonra: \n {df_clean}") #Deneyim yılı 15 olan outlier silindi.

#Label Encoding ve One-Hot Encoding

label_encoder = LabelEncoder()

# Hedef Değişkeni Sayısal Hale Getir
y = label_encoder.fit_transform(df_clean["satin_aldi"])

print(f"Hedef değişken sınıfı: \{label_encoder.classes_}")
print(y)

# Hedef Sutunu Veri Setinden Çıkart.
x = df_clean.drop(columns=["satin_aldi"])

x = pd.get_dummies(x,columns=["egitim"], drop_first=True, dtype=int)

print(f"Kategorik dönüşümün sonrası özellikler: n\ {x}")

# Veriyi Train Validation ve Test Kümelerine Ayır.
x_train_val, x_test, y_train_val, y_test = train_test_split(x,y,test_size=0.2, random_state = 42, stratify=y) #val 80 test 

x_train, x_val, y_train, y_val = train_test_split(x_train_val,y_train_val, test_size=0.4,random_state=42, stratify=y_train_val)

print(f"X_Train : {x_train.shape}")
print(f"X_Val : {x_val.shape}")
print(f"X_Test : {x_test.shape}")

# Sayisal Özelliklerde Standartization

standard_Scaler = StandardScaler()

x_train_standard = x_train.copy()
x_val_standard = x_val.copy()
x_test_standard = x_test.copy()

# Ölçekleyiciyi yalnızca eğitim verisi üzerinde öğretiyoruz.

x_train_standard[sayisal_Sutunlar] = (
    standard_Scaler.fit_transform(
        x_train[sayisal_Sutunlar]
    )
)
# Validasyon ve test verilerinde yalnızca transform uygula.

x_val_standard[sayisal_Sutunlar] = (
    standard_Scaler.transform(
        x_val[sayisal_Sutunlar]
    )
)

x_test_standard[sayisal_Sutunlar] = (
    standard_Scaler.transform(
        x_test[sayisal_Sutunlar]
    )
)

print(f"X_train_standard: \n{x_train_standard}")

# 8. normalizasyon

minmax_scaler = MinMaxScaler()

X_train_normalized = x_train.copy()
X_val_normalized = x_val.copy()
X_test_normalized = x_test.copy()

# ölçekleyiciyi yalnızca eğitim verisi üzerinde öğretiyoruz
X_train_normalized[sayisal_Sutunlar] = (
    minmax_scaler.fit_transform(
        x_train[sayisal_Sutunlar]
    )
)

# validasyon ve test verilerinde yalnızca transform uygula
X_val_normalized[sayisal_Sutunlar] = (
    minmax_scaler.transform(
        x_val[sayisal_Sutunlar]
    )
)

X_test_normalized[sayisal_Sutunlar] = (
    minmax_scaler.transform(
        x_test[sayisal_Sutunlar]
    )
)

print(f"X_train_normalized: \n{X_train_normalized}")
