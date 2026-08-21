"""
Amaç:
    1-K-Fold, Stratified K-Fold ve Leave-One Out yöntemlerinin uygulanması
    2-Bu üç yöntemin model değerlendirme mantığının sade ve karşılaştırmalı olarak göster
Adımlar:
    1. gerekli kütüphanelerin içeriye aktarılması
    2. örnek veri setini yükle
    3. basit bir sınıflandırma modeli tanımla
    4. K-fold ile çapraz doğrulama yapalım
    5. Stratified K-fold ile çapraz doğrulama yapılması
    6. Leave-one-out ile çapraz doğrulama yapılması
    7. Sonuçların birlikte yazdırılması

Kurulumlar:
pip install scikit-learn numpy
"""
# 1. gerekli kütüphanelerin içeriye aktarılması
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score

# 2. örnek veri setini yükle
X, y = load_iris(return_X_y=True)

# 3. basit bir sınıflandırma modeli tanımla
model = DecisionTreeClassifier(random_state=42,max_depth=5)

# 4. K-fold ile çapraz doğrulama yapalım
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_accuracy = cross_val_score(model, X, y, cv = kfold, scoring="accuracy")

# 5. Stratified K-fold ile çapraz doğrulama yapılması
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_kfold_accuracy = cross_val_score(model, X, y, cv = stratified_kfold, scoring="accuracy")

# 6. Leave-one-out ile çapraz doğrulama yapılması
loo = LeaveOneOut()
loo_accuracy = cross_val_score(model, X, y, cv = loo, scoring="accuracy")

# 7. Sonuçların birlikte yazdırılması
print(f"kfold: {kfold_accuracy}")
print(f"stratified_kfold: {stratified_kfold_accuracy}")
print(f"loo_accuracy: {loo_accuracy}")

print("kfold")
print(np.mean(kfold_accuracy))
print(np.std(kfold_accuracy))

print("stratified_kfold")
print(np.mean(stratified_kfold_accuracy))
print(np.std(stratified_kfold_accuracy))

print("loo")
print(np.mean(loo_accuracy))
print(np.std(loo_accuracy))

"""
kfold:            
[1.         0.96666667 0.93333333 0.93333333 0.93333333]

stratified_kfold:
[0.96666667 0.96666667 0.93333333 0.96666667 0.9       ]

loo_accuracy:
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1.
 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1.]

kfold
0.9533333333333335
0.02666666666666666

stratified_kfold
0.9466666666666667
0.02666666666666666

loo
0.94
0.23748684174075832

"""