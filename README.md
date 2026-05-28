# Papilödem Sınıflandırması — Radyomik Özelliklerle Makine Öğrenmesi

Üsküdar Üniversitesi Fen Bilimleri Enstitüsü  
**Ders:** Makine Öğrenmesi  
**Ödev Türü:** Final Projesi  
**Problem:** İkili Sınıflandırma (Normal vs Papilödem)  
**GitHub:** https://github.com/barisaykent-spec/papilodema-classification

---

## Proje Özeti

Bu projede 746 radyomik özellik içeren göz muayenesi verisi kullanılarak papilödem tespiti için makine öğrenmesi pipeline'ı geliştirilmiştir. 69 hastadan elde edilen 966 ROI bileşeni analiz edilmiştir.

| Parametre | Değer |
|---|---|
| Toplam hasta | 69 (48 Normal, 21 Papilödem) |
| Toplam satır | 966 |
| Özellik sayısı | 746 |
| Problem türü | İkili sınıflandırma |
| Hasta başına satır | 14 |

---

## Klasör Yapısı

```
papilodema_classification/
│
├── data/                    # Ham veri (CSV dosyaları)
│   ├── normal_radiomics.csv
│   └── papilodem_radiomics.csv
│
├── notebooks/               # Jupyter Notebook'lar (sıralı)
│   ├── 01_veri_kesfi.ipynb
│   ├── 02_derin_eda.ipynb
│   ├── 03_on_isleme.ipynb
│   ├── 04_mrmr_ozellik_secimi.ipynb
│   ├── 05_model_egitimi_optuna.ipynb
│   ├── 06_ensemble_kalibrasyon.ipynb
│   └── 07_performans_ve_grafikler.ipynb
│
├── figures/                 # Üretilen grafikler
├── models/                  # Kaydedilen modeller (.pkl)
├── results/                 # Sonuç tabloları (.csv)
└── report/                  # Final raporu (.pdf)
```

---

## Pipeline Adımları

1. **Veri Keşfi (EDA)** — Hasta yapısı, sınıf dengesi, özellik dağılımları
2. **Derin EDA** — Mann-Whitney testi, PCA, korelasyon, varyans analizi
3. **Ön İşleme** — Median imputation, low-variance filtering, korelasyon eleme, RobustScaler
4. **Özellik Seçimi** — MRMR (Minimum Redundancy Maximum Relevance)
5. **Hiperparametre Optimizasyonu** — Optuna, TPE sampler, 50+ trial
6. **Model Eğitimi** — Random Forest, Extra Trees, Gradient Boosting
7. **Kalibrasyon** — Sigmoid calibration
8. **Ensemble** — Soft Voting (RF + ET + GB)
9. **Değerlendirme** — ROC-AUC, PR-AUC, F1, Brier Score, ECE-10
10. **İstatistiksel Analiz** — Friedman, Wilcoxon, Bonferroni

---

## Kurulum

```bash
pip install pandas numpy scikit-learn optuna matplotlib seaborn scipy xgboost lightgbm mrmr-selection
```

---

## Notlar

- Tüm ön işleme adımları yalnızca eğitim verisine fit edilmiştir (veri sızıntısı yok).
- Hasta bazında veri bölme uygulanmıştır (aynı hastanın satırları farklı setlere düşmez).
- Test seti yalnızca final değerlendirmede kullanılmıştır.
