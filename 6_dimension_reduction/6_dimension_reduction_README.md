# 6. Dimension Reduction (Boyut İndirgeme)

Bu bölümde, yüksek boyutlu veriyi görselleştirilebilir/işlenebilir daha düşük boyuta indirgeme yöntemlerini uyguluyorum: **PCA** ve **t-SNE**.

## İçerik

### `dimension_reduction.py`

Iris veri seti (150 örnek, 4 öznitelik, 3 sınıf) üzerinde iki farklı boyut indirgeme yönteminin karşılaştırmalı uygulaması — 4 boyuttan 2 boyuta indirgeme:

1. **Veri Yükleme** — Iris veri setinin yüklenmesi
2. **Özellik/Hedef Ayrımı** — X (4 öznitelik) ve y (tür/sınıf etiketi) ayrıldı
3. **Standartlaştırma** — Veriler `StandardScaler` ile ölçeklendirildi (PCA ve t-SNE, öznitelikler farklı ölçeklerde olduğunda yanlı sonuç verebileceği için standartlaştırma öncesi gereklidir)
4. **PCA (Principal Component Analysis)**
   - `n_components=2` ile model tanımlandı
   - Veri, varyansı en çok koruyan 2 doğrusal bileşene (principal component) indirgendi
   - Sonuç scatter plot ile görselleştirildi (sınıflara göre renklendirilerek)
5. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
   - `n_components=2` ile model tanımlandı
   - Veri noktaları arası yerel komşuluk ilişkilerini koruyacak şekilde doğrusal olmayan bir indirgeme yapıldı
   - Sonuç scatter plot ile görselleştirildi

## Kullanılan Kütüphaneler

- scikit-learn (`PCA`, `TSNE`, `StandardScaler`)
- matplotlib

## Notlar

- **PCA** doğrusal bir yöntemdir, hızlıdır ve bileşenlerin açıkladığı varyans oranı yorumlanabilir; ancak doğrusal olmayan yapıları (manifold) yakalamakta sınırlıdır.
- **t-SNE** doğrusal olmayan bir yöntemdir, özellikle kümelerin görsel ayrımını PCA'dan daha net gösterebilir; ancak hesaplama maliyeti daha yüksektir, sonuçlar rastgele başlangıca (`random_state`) duyarlıdır ve eksenlerin/mesafelerin doğrudan yorumlanabilir bir anlamı yoktur (sadece komşuluk ilişkisini korur).
- Bu iki yöntem farklı amaçlarla kullanılır: PCA genelde ön işleme/gürültü azaltma için, t-SNE ise çoğunlukla görselleştirme/keşifsel analiz için tercih edilir.
