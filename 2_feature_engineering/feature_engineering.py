"""
Öznitelik Mühendisliği

Amaç:
    1.Mevcut sütunlarda yeni öznitelik üretme mantığını basit bir örnek ile uygulama
    2.Korelasyon üzerinden modele daha faydalı olabilecek öznitelikleri seçme mantığını gösterme
Adımlar:
    1.Gerekli kütüphanelerin içeriye aktarılması
    2.Veri setini yükleme
    3.Mevcut sütunlardan yeni öznitelikler üretmek(feature extraction)
    4.Hedef değişken ile öznitelikler arasındaki korelasyonları inceleme
    5.Yüksek korelaseyon değerine göre yüksek olan özniteliklerin seçilmesi(feature selection)

Kurulumlar
pip install pandas
"""

#1.Gerekli kütüphanelerin içeri aktarılması

import pandas as pd

#2.Veri setinin yüklenmesi
df = pd.read_csv("oznitelik_muhendisligi_pratik.csv")
print(df)

#3.Mevcut sütunlardan yeni öznitelikler üretmek(feature extraction)
df["deneyim_orani"] = df["deneyim_yili"] / df["yas"]
df["yillik harcama tahmini"] = df["aylik_harcama"] * 12

print(df.head())

#4.Hedef değişken ile öznitelikler arasındaki korelasyonları inceleme
sayisal_df = df.drop("sehir", axis=1)
korelasyonlar = sayisal_df.corr(numeric_only=True)["performans_puani"].sort_values(ascending=False)
print(korelasyonlar)

"""
performans_puani          1.000000
deneyim_orani             0.821244(yüksek pozitif korelasyob)
deneyim_yili              0.597232(orta-yüksek)
yillik harcama tahmini    0.317301(orta)
aylik_harcama             0.317301(orta)
yas                      -0.224902(orta-dusuk negatif)
uyelik_suresi_ay         -0.238212(orta-dusuk negatif)
"""

#5.Yüksek korelaseyon değerine göre yüksek olan özniteliklerin seçilmesi(feature selection)
secilen_oznitelikler = korelasyonlar[abs(korelasyonlar)> 0.75].index.tolist()
secilen_oznitelikler.remove("performans_puani")
print(secilen_oznitelikler)