# ML Course Notes

Türkiye Yapay Zeka Akademisi Makine Öğrenmesi kursu kapsamında yazdığım kod ve notlar. Kurs ilerledikçe yeni klasörler eklenerek büyütülecektir.

## İçerik

| Klasör | Konu |
|---|---|
| `1_data_preprocessing` | Eksik veri tespiti/doldurma, IQR ile aykırı değer tespiti, label/one-hot encoding, train-test-validation split, standardization/normalization |
| `2_feature_engineering` | Yeni öznitelik üretme (feature extraction), korelasyon bazlı öznitelik seçimi (feature selection) |
| `3_supervised_learning` | Logistic Regression (UCI Heart Disease), Linear/Polynomial/Lasso/Ridge Regression (sentetik veri), KNN (Breast Cancer), SVM (Digits), Decision Tree & Random Forest (Iris) |
| `4_customer_churn_pred` | Telco Customer Churn — uçtan uca sınıflandırma projesi: preprocessing, encoding, 5-fold CV ile model karşılaştırması (Logistic Regression, KNN, Decision Tree, Random Forest, SVM), final değerlendirme |
| `5_unsupervised_learning` | K-Means & Agglomerative Clustering (sentetik veri, dendrogram) — devam ediyor |

Her klasörün içinde, o bölümdeki kodların amacını ve adımlarını anlatan ayrı bir `README.md` bulunuyor.

## Kullanılan Araçlar

- Python
- pandas, NumPy
- scikit-learn
- scipy
- matplotlib / seaborn

## Notlar

Bu repo öğrenme sürecimin bir kaydıdır; kod parçaları kurs alıştırmalarını ve kişisel deneylerimi içerir. İlerleyen bölümler (model değerlendirme, hyperparameter tuning, SHAP/LIME vb.) eklendikçe güncellenecektir.
