# 5. Unsupervised Learning

Bu bölümde, etiketlenmemiş veri üzerinde yapı/örüntü bulmaya yönelik kümeleme (clustering) algoritmalarını uyguluyorum. Kurs devam ettikçe bu klasöre yeni konular (boyut indirgeme: PCA, t-SNE vb.) eklenecek.

> **Durum:** Devam ediyor. Şu ana kadar KMeans ve Agglomerative (Hiyerarşik) Clustering işlendi. Boyut indirgeme konusuna henüz başlanmadı.

## İçerik

### `clustering.py`

Sentetik veri (`make_blobs`) üzerinde iki farklı kümeleme algoritmasının karşılaştırmalı uygulaması:

1. **Veri oluşturma & görselleştirme** — `make_blobs` ile 4 merkezli, 300 noktalık 2 boyutlu sentetik veri seti üretimi ve ham verinin scatter plot ile incelenmesi.

2. **K-Means Clustering**
   - `n_clusters=4`, `n_init=10`, `random_state=42` parametreleriyle model eğitimi
   - Merkez bazlı (centroid-based) kümeleme: her nokta en yakın merkeze atanır, merkezler iteratif olarak güncellenir
   - Öğrenilen küme etiketlerinin (`labels_`) ve küme merkezlerinin (`cluster_centers_`) görselleştirilmesi

3. **Agglomerative (Hiyerarşik) Clustering**
   - `n_clusters=4`, `linkage="ward"` parametreleriyle model eğitimi
   - Bottom-up yaklaşım: her nokta kendi kümesiyle başlar, en yakın kümeler adım adım birleştirilir
   - K-Means'ten farkı: rastgele başlangıca duyarlı değil (deterministik), merkez kavramı yok
   - Sonuç kümelerinin scatter plot ile görselleştirilmesi

4. **Dendrogram**
   - `scipy.cluster.hierarchy.linkage` ve `dendrogram` ile hiyerarşik birleştirme sürecinin tamamının ağaç diyagramı olarak çizilmesi
   - Küme sayısına karar vermek için görsel bir araç: dikey eksendeki en uzun kesintisiz boşluk, "doğal" küme sayısını gösterir

## Kullanılan Kütüphaneler

- scikit-learn (`KMeans`, `AgglomerativeClustering`)
- scipy (`scipy.cluster.hierarchy` — `linkage`, `dendrogram`)
- matplotlib

## Notlar

- `KMeans` küresel (spherical) küme şekli varsayımıyla çalışır ve rastgele başlangıca duyarlıdır (`n_init` ile bu risk azaltılır); `AgglomerativeClustering` ise deterministiktir ve merkez kavramı olmadığı için farklı şekilli kümelerde bazen daha esnek sonuç verebilir.
- Bu örnekte küme sayısı (`n_clusters=4`) veri üretilirken zaten bilindiği için sabit verildi. Gerçek/etiketsiz veride bu sayıya Elbow Method, Silhouette Score veya dendrogram incelemesiyle karar verilir — bu konu ileride eklenecek.
