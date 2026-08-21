7. Cross Validation (Çapraz Doğrulama)
Bu bölümde, model performansını tek bir train/test bölümüne bağlı kalmadan daha güvenilir şekilde değerlendirmek için üç farklı çapraz doğrulama yöntemini karşılaştırıyorum: K-Fold, Stratified K-Fold ve Leave-One-Out (LOOCV).
İçerik
`cross_validation.py`
Iris veri seti (150 örnek, 4 öznitelik, 3 sınıf) üzerinde Decision Tree sınıflandırıcısı ile üç CV yönteminin karşılaştırmalı uygulaması:

1. Veri Yükleme — Iris veri setinin yüklenmesi
2. Model Tanımlama — Derinliği kısıtlanmış (`max_depth=5`) bir Decision Tree tanımlandı
3. K-Fold — Veri 5 eşit parçaya (fold) rastgele bölünerek çapraz doğrulama yapıldı
4. Stratified K-Fold — Her fold'da sınıf oranlarının korunduğu çapraz doğrulama yapıldı
5. Leave-One-Out — Her örneğin sırayla tek başına test verisi olduğu, n adet fold içeren çapraz doğrulama yapıldı
6. Üç yöntemin mean/std accuracy skorları karşılaştırıldı

Kullanılan Kütüphaneler

* scikit-learn (`DecisionTreeClassifier`, `KFold`, `StratifiedKFold`, `LeaveOneOut`, `cross_val_score`)
* numpy

Notlar

* Dengeli veri setlerinde (iris gibi) K-Fold ve Stratified K-Fold sonuçları birbirine yakın çıkar; Stratified'in asıl avantajı sınıf dağılımı dengesiz olan veri setlerinde ortaya çıkar.
* LOOCV, her örneği ayrı ayrı test ettiği için en düşük bias'lı performans tahminini verir, ancak standart sapması diğer yöntemlerle doğrudan kıyaslanamaz (0/1 skorlardan hesaplandığı için farklı bir varyans kaynağını ölçer) ve hesaplama maliyeti büyük veri setlerinde yüksektir.
* CV yöntemi seçimi, veri seti büyüklüğü, sınıf dengesi ve hesaplama bütçesine göre yapılmalıdır.
