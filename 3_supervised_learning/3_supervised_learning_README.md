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

## SVM (Support Vector Machine)

**Amaç:** Digits veri seti kullanarak SVM ile çok sınıflı bir sınıflandırma problemi çözmek.

**Veri Seti:** Digits veri seti, 0-9 arasındaki rakamları temsil eden 8x8 boyutunda gri seviyeli görüntülerden oluşur. 1797 örnek (sample) içerir; her örnekte 8x8 pikselden gelen 64 öznitelik (feature) vardır.

**Plan:**
1. Veri setinin yüklenmesi ve temel bilgilerin incelenmesi
2. Örnek görüntülerin görselleştirilmesi
3. Özellik ve hedef değişkenlerin ayrılması
4. Eğitim ve test veri setlerinin oluşturulması
5. SVM modelinin oluşturulması
6. Modelin eğitilmesi
7. Test verisi üzerinde tahmin yapılması
8. Model performansının sınıflandırma raporu ile değerlendirilmesi

## Decision Tree & Random Forest

**Amaç:**
1. Iris veri seti kullanarak karar ağacı ve random forest algoritmalarını geliştirmek
2. Karar ağacını görselleştirmek ve öznitelik önemini (feature importance) incelemek

**Veri Seti:** Iris veri seti, 3 farklı çiçek türünü içerir (setosa, versicolor, virginica). 4 öznitelik (sepal length, petal length, sepal width, petal width) ve 150 örnek içerir.

**Plan:**
1. Veri setinin yüklenmesi ve incelenmesi
2. Feature ve target değişkenlerin tanımlanması
3. Eğitim ve test veri setlerinin oluşturulması
4. Karar ağacı ve random forest modellerinin oluşturulması
5. Test verisi ile tahmin yapılması
6. Model başarımının accuracy ile ölçülmesi
7. Karar ağacı sonuçlarının confusion matrix ile görselleştirilmesi
8. Karar ağacının görselleştirilmesi
9. Karar ağacı feature importance incelenmesi

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

