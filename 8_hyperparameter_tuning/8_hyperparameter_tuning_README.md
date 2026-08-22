# 7. Hyperparameter Tuning

Bu klasör, farklı sınıflandırma modelleri için hiperparametre optimizasyonunun `GridSearchCV` ve `RandomizedSearchCV` ile nasıl yapıldığını, sonuçların nasıl karşılaştırılacağını ve yorumlanacağını kapsar.

## İçerik

- `sklearn.Pipeline` kullanarak ön işleme + model adımlarını tek bir nesnede birleştirme
- `GridSearchCV`: tanımlı hiperparametre uzayının tamamını (exhaustive) tarama
- `RandomizedSearchCV`: uzaydan rastgele örnekleme yaparak daha hızlı arama
- Cross-validation (`cv`) ile CV skoru üretme ve bunu ayrı tutulan test seti skoruyla karşılaştırma
- 3 farklı modelin (KNN, Decision Tree, Logistic Regression) aynı çerçevede karşılaştırılması

## Neden Pipeline?

Ön işleme adımlarını (örn. scaling) model ile aynı `Pipeline` içine koymak, cross-validation sırasında **veri sızıntısını (data leakage)** engeller. Her CV fold'unda ön işleme adımı yalnızca o fold'un eğitim kısmına `fit` edilir, doğrulama kısmına sadece `transform` uygulanır — böylece gerçek dünyadaki (production) davranış simüle edilmiş olur.

Pipeline içindeki bir adımın hiperparametresine erişmek için `<adım_ismi>__<parametre_ismi>` (çift alt çizgi) sözdizimi kullanılır. Örnek: `model__n_neighbors`.

## Grid Search vs. Random Search

| | Grid Search | Random Search |
|---|---|---|
| Arama şekli | Tanımlı tüm kombinasyonları dener | Uzaydan rastgele `n_iter` kadar örnek dener |
| Küçük/sabit uzaylarda | Daha güvenilir (tüm uzayı tarar) | Kombinasyon kaçırma riski var |
| Büyük/sürekli uzaylarda | Yavaş, hesaplama maliyeti yüksek | Çok daha hızlı, "yeterince iyi" sonuç |
| Bu klasördeki kurulum | `param_grid` ile sabit, küçük liste | Aynı sabit listeden `n_iter=4` örnek |

> Not: Bu denemede `random_params`, `grid_params` ile aynı (sonlu) uzayı paylaştığı için Random Search'ün asıl avantajı (sürekli dağılımlarla geniş uzay taraması) gözlemlenmedi. Random Search'ün gücünü görmek için `scipy.stats` ile sürekli dağılımlar (`loguniform`, `randint` vb.) tanımlanıp `n_iter` artırılmalı.

## Deneyde Kullanılan Modeller

- **KNN** (`n_neighbors`, `metric`)
- **Decision Tree** (`max_depth`, `min_samples_split`, `criterion`)
- **Logistic Regression** (`C`, `penalty`)

## Sonuç Özeti

| Model | Yöntem | CV Skoru | Test Skoru |
|---|---|---|---|
| KNN | Grid | 0.95 | 0.94 |
| KNN | Random | 0.95 | 0.93 |
| Decision Tree | Grid | 0.94 | 0.94 |
| Decision Tree | Random | 0.93 | 0.91 |
| Logistic Regression | Grid | 0.96 | **0.98** |
| Logistic Regression | Random | 0.95 | 0.96 |

**Çıkarımlar:**
- Logistic Regression, hem CV hem test setinde en iyi performansı verdi — veri setindeki sınıfların büyük ölçüde doğrusal olarak ayrılabilir olduğuna işaret ediyor.
- Grid Search, sabit/küçük arama uzayı nedeniyle her modelde Random Search'e eşit ya da üstün sonuç verdi.
- Decision Tree'nin Random Search sonucunda CV-test farkının diğer modellere göre biraz daha büyük olması, karar ağaçlarının overfitting'e yatkınlığıyla tutarlı.

## Sonraki Adım Fikri

`random_params` içine sürekli dağılımlar tanımlayıp (`scipy.stats.loguniform`, `randint`) `n_iter`'ı artırarak Grid Search ile Random Search arasındaki gerçek performans/hız farkını daha büyük bir uzayda test etmek.
