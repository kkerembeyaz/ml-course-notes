# Supervised Learning

Regresyon ve sınıflandırma algoritmalarının farklı veri setleri üzerinde uygulanması.

## Logistic Regression

**Amaç:** UCI Heart Disease veri setini kullanarak lojistik regresyon modeli ile ikili sınıflandırma problemi çözmek. Model, bir bireyin kalp hastalığına sahip olup olmadığını tahmin eder ve accuracy metriği ile değerlendirilir.

**Veri Seti:** [UCI ML Repo — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease). Bireylere ait demografik ve klinik ölçümler içerir (Yaş, Cinsiyet, Ağrı Tipi, Kolesterol, Kan Basıncı vb.)

**Plan:**
- Veri setini yükle, temel analizleri yap
- Eksik değer kontrolü yap, gerekirse temizle
- Öznitelik ve hedef değişkenleri ayır
- Eğitim ve test veri setlerini oluştur
- Lojistik regresyon modelini tanımla ve eğit
- Modeli test veri seti ile değerlendir

## Linear / Polynomial / Lasso / Ridge Regression

**Amaç:**
1. Sentetik veri seti oluşturmak
2. Doğrusal, polinomal, lasso ve ridge regresyon modellerini uygulamak
3. Lasso ile feature selection yapmak

**Adımlar:**
1. Gerekli kütüphanelerin içe aktarılması
2. Sentetik veri seti oluşturulması
3. Oluşturulan verinin görselleştirilmesi
4. Veriyi eğitim/test olarak ayırma
5. Doğrusal, polinomal, lasso ve ridge regresyon modellerinin oluşturulması
6. Modellerin eğitimi (training) ve tahmini (prediction)
7. Modellerin performans karşılaştırması
8. Lasso ile feature selection

## KNN (K-Nearest Neighbors)

**Amaç:** Göğüs kanseri (Breast Cancer) veri setini kullanarak KNN algoritması ile sınıflandırma yapmak. Modelin doğruluk oranını hesaplamak ve farklı K değerleri için hiperparametre araması yapmak.

**Plan:**
- Veri setinin yüklenmesi
- Feature ve hedef değişkenlerin ayrılması
- Eğitim ve test verilerinin oluşturulması
- Özelliklerin ölçeklendirilmesi
- KNN eğitimi ve testi
- Doğruluk oranı ve confusion matrix
- Hiperparametre ayarlanması (K değeri optimizasyonu)
- Sonuçların grafiksel gösterilmesi
