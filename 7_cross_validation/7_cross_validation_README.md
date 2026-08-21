# Cross Validation (Çapraz Doğrulama)

Bu bölümde, model performansını daha güvenilir şekilde değerlendirmek için üç farklı çapraz doğrulama (cross-validation) yöntemini karşılaştırmalı olarak uyguluyorum: K-Fold, Stratified K-Fold ve Leave-One-Out (LOOCV).

## İçerik

`cross_validation.py`

Iris veri seti (150 örnek, 4 öznitelik, 3 sınıf) üzerinde Decision Tree sınıflandırıcısı kullanılarak üç CV yönteminin karşılaştırmalı uygulaması:

1. **Veri Yükleme** — Iris veri setinin yüklenmesi
2. **Model Tanımlama** — `DecisionTreeClassifier(random_state=42, max_depth=5)` ile aşırı öğrenmeyi (overfitting) sınırlamak amacıyla derinliği kısıtlanmış bir ağaç tanımlandı
3. **K-Fold CV** — `KFold(n_splits=5, shuffle=True, random_state=42)` ile 5 katlı çapraz doğrulama
4. **Stratified K-Fold CV** — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ile sınıf oranlarını her fold'da koruyan çapraz doğrulama
5. **Leave-One-Out CV** — `LeaveOneOut()` ile her örneğin sırayla test verisi olduğu, toplam 150 fold'luk çapraz doğrulama
6. Her yöntem için `cross_val_score` ile accuracy skorları hesaplandı, ortalama (mean) ve standart sapma (std) karşılaştırıldı

## Sonuçlar

| Yöntem | Mean Accuracy | Std | Fold Sayısı |
|---|---|---|---|
| K-Fold | 0.9533 | 0.0267 | 5 |
| Stratified K-Fold | 0.9467 | 0.0267 | 5 |
| LOOCV | 0.9400 | 0.2375 | 150 |

## Kullanılan Kütüphaneler

* scikit-learn (`DecisionTreeClassifier`, `KFold`, `StratifiedKFold`, `LeaveOneOut`, `cross_val_score`)
* numpy

## Notlar

* Iris veri seti sınıf dağılımı açısından dengeli olduğu için (her sınıftan 50 örnek), K-Fold ile Stratified K-Fold arasındaki fark burada ihmal edilebilir düzeydedir. Stratified'in asıl avantajı dengesiz (imbalanced) veri setlerinde ortaya çıkar.
* LOOCV'de her fold tek bir örneği test ettiği için skorlar yalnızca 0 veya 1 olabilir; bu yüzden LOOCV'nin standart sapması, K-Fold/Stratified K-Fold'un standart sapmasıyla aynı ölçekte değildir ve doğrudan karşılaştırılamaz — farklı bir varyans kaynağını (örnek bazlı vs. fold bazlı) ölçer.
* LOOCV en düşük bias'a sahip tahmin yöntemidir çünkü eğitim verisini maksimum düzeyde kullanır, ancak hesaplama maliyeti (n adet ayrı model eğitimi) büyük veri setlerinde pratik değildir.
* `random_state` hem modelde hem de fold bölme işlemlerinde sabitlendi; bu, sonuçların tekrarlanabilir (reproducible) olmasını sağlar.
