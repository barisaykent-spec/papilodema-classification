# Makine Öğrenmesi Final Projesi — Öğrenme Dökümanı
### Papilödema Sınıflandırma | Barış Refik Aykent | 244329058

---

## Bu Döküman Ne İçin?

Bu proje boyunca 8 notebook yazdık, 3 bonus analiz ekledik, PDF ve Word rapor hazırladık.  
Ama "neden böyle yaptık?" sorusunun cevabı dağınık kaldı.  
Bu döküman o soruyu cevaplıyor — kodu değil, **mantığı** öğretmek için.

---

## 🗺️ Büyük Resim — Ne İnşa Ettik?

```
Ham Görüntü (Göz Fundusu)
        ↓
   Radiomik Özellikler (746 adet sayısal değer)
        ↓
   Ön İşleme (temizlik, ölçekleme)
        ↓
   MRMR Özellik Seçimi (746 → 10)
        ↓
   Optuna (en iyi parametreleri bul)
        ↓
   RF + ET + GB → Soft Voting Ensemble
        ↓
   Karar: Normal mi? Papilödema mı?
```

**Neden bu sıra?** Çünkü her adım bir öncekinin çıktısını alıyor.  
Ham görüntüden direkt model kuramazsın — önce sayısallaştırman, temizlemen, en önemli özellikleri seçmen lazım.

---

## 📦 Veri — Ne Vardı Elimizde?

| Özellik | Değer |
|---------|-------|
| Toplam hasta | 69 |
| Normal | 48 |
| Papilödema | 21 |
| Görüntü başına örnek | ~5-10 |
| Toplam radiomik özellik | 746 |

**Önemli Problem:** Normal hastalar 1-48, Papilödema hastaları da 1-21 numaralıydı.  
Yani iki farklı listede aynı numaralar vardı. Bunu düzeltmezsen model karışır.  
**Fix:** Papilödema hastaların numarasına 1000 ekledik → artık çakışma yok.

```python
papil['PatientIndex'] += 1000  # 1→1001, 2→1002 ...
```

**Neden bu önemli?** Hasta bazlı bölme yapıyoruz. Aynı hastanın görüntüleri hem train'e hem test'e girerse model "ezberler", gerçekte başarısız olur.

---

## 🔧 Ön İşleme — Neden Gerekli?

Ham radiomik değerler kirli gelir: eksik değerler, çok büyük/küçük sayılar, anlamsız değişkenler.  
Biz 4 adımlı bir pipeline kurdu:

### Adım 1: Eksik Değer Doldurma (Median Imputation)
```
Özellik X: [1.2, NaN, 3.4, 2.1, NaN]
Medyan: 2.1
Sonuç:  [1.2, 2.1, 3.4, 2.1, 2.1]  ✓
```
Neden medyan? Ortalama aykırı değerlerden etkilenir, medyan etkilenmez.

### Adım 2: Varyans Filtresi
Hiç değişmeyen özellikler model için işe yaramaz.  
Tüm hastalarda aynı değeri gösteren özelliği sil → `VarianceThreshold(0.01)`

### Adım 3: Korelasyon Filtresi
İki özellik birbirine çok benziyorsa (korelasyon > 0.95) birini sil.  
"Aynı şeyi iki kez söyleme" prensibi. 746 → ~200-300 özelliğe indi.

### Adım 4: RobustScaler
Özelliklerin ölçeğini eşitle. Biri 0-1 arasında, diğeri 0-1000 arasında olursa model büyük olanı önemli sanır.  
RobustScaler medyan ve IQR kullanır → aykırı değerlerden etkilenmez.

---

## 🎯 MRMR Özellik Seçimi — 746'dan 10'a Nasıl?

**Problem:** 746 özellik çok fazla. Model ezberlemeye başlar, genelleşemez.  
**Çözüm:** En bilgi taşıyan, birbirinden bağımsız 10 özelliği seç.

**MRMR = Maximum Relevance, Minimum Redundancy**

```
Maximum Relevance → Hedefle (Normal/Papilödema) yüksek korelasyon
Minimum Redundancy → Seçilen özellikler birbirinden farklı olsun
```

**Finans analojisi:** Portföy çeşitlendirmesi gibi.  
Hepsi aynı sektörden hisse alırsan (yüksek korelasyon) risk azalmıyor.  
Farklı sektörlerden al (düşük korelasyon) → gerçek çeşitlendirme.

**Neden 10?** Farklı K değerlerini denedik (5, 8, 10, 15, 20).  
10'da AUC en yüksekti, fazlası artık değer katmıyordu.

---

## 🌳 Modeller — RF, ET, GB Ne Fark Eder?

### Random Forest (RF)
- 100-200 bağımsız ağaç kur
- Her ağaç rastgele hasta grubuyla, rastgele özelliklerle öğrenir
- Hepsini oylat → çoğunluk kazanır
- **Güçlü:** Genel problemlerde stabil

### Extra Trees (ET)
- RF'ye çok benzer ama dal ayrım noktaları da rastgele seçilir
- Daha "dağınık" düşünür → gürültülü veride iyi
- **Güçlü:** Radiomik veriler gürültülüdür, ET burada işe yarar

### Gradient Boosting (GB)
- Ağaçlar sıralı eklenir, her biri öncekinin hatasını düzeltir
- Finans analojisi: Risk yönetimi — her adımda hatayı minimize et
- **Güçlü:** Yüksek doğruluk, ama yavaş ve aşırı öğrenmeye meyilli

### Neden Üçünü Birleştirdik?
```
Hasta 7:
  RF  → "Normal"      ❌
  ET  → "Papilödema"  ✅
  GB  → "Papilödema"  ✅
Karar: 2-1 → Papilödema  ✅
```
Tek model yanıldığında diğerleri kurtarıyor. **Soft Voting** ile olasılıkların ortalaması alınıyor, bu ham oy sayısından daha iyi çalışır.

---

## ⚙️ Optuna — Neden Elle Parametr Seçmedik?

Her modelin onlarca parametresi var:
- Kaç ağaç? (50 mi, 100 mü, 500 mü?)
- Maksimum derinlik? (3 mü, 5 mi, sınırsız mı?)
- Minimum yaprak boyutu?

Bunları elle denemek → binlerce kombinasyon.  
**Optuna** TPE algoritmasıyla akıllıca arama yapar — iyi sonuç veren bölgeye odaklanır, kötü bölgeleri eler.  
50 deneme yaptı, her modelin en iyi parametresini buldu.

---

## 📊 Performans Metrikleri — Hangisi Neden Önemli?

| Metrik | Formül | Bizim İçin Önemi |
|--------|--------|------------------|
| **Accuracy** | Doğru / Toplam | Genel başarı — ama dengesiz veride yanıltır |
| **Sensitivity (Recall)** | TP / (TP+FN) | Hasta olanı kaçırmama — tıbbi açıdan kritik |
| **Specificity** | TN / (TN+FP) | Sağlıklıyı yanlış hasta saymama |
| **F1 Macro** | Harmonic mean | Her iki sınıf için dengeli başarı |
| **ROC-AUC** | Eğri altı alan | Eşik bağımsız genel performans |
| **PR-AUC** | Precision-Recall | Dengesiz veri için ROC'tan daha güvenilir |

**Neden Accuracy tek başına yetmez?**  
21 Papilödema, 48 Normal var. Model hepsine "Normal" dese bile Accuracy = %70 görünür.  
Ama tüm hastaları kaçırdın — Sensitivity = 0.  
Bu yüzden F1 ve AUC daha anlamlı.

---

## 🔁 20 Tekrarlı Hasta Bazında Bölme — Neden?

**Tek split problemi:** Bir kere böldük, şans eseri test grubuna kolay hastalar düştü → yüksek skor ama gerçek değil.

**Çözüm:** 20 farklı rastgele bölme yap, her birinde sonucu ölç, **ortalamasını** al.

```python
random_state = split_no * 7 + 42  # Her seferinde farklı bölme
```

**Neden hasta bazlı?** Aynı hastanın 5 görüntüsü var. Görüntü bazlı böldüğünde aynı hastanın bazı görüntüleri train'e, bazıları test'e girebilir — model o hastayı "tanımış" olur. Hasta bazlı bölmede bir hasta ya tamamen train'de ya da tamamen test'te.

**Sonuç:** `Macro F1: 0.87 ± 0.096` — ortalama 0.87, standart sapma 0.096.  
Yüksek sapma = model kararsız. Bizimki kabul edilebilir aralıkta.

---

## 📈 İstatistiksel Testler — Neden Sadece Skor Vermek Yetmez?

"RF'nin F1'i 0.87, ET'nin 0.86 — RF daha iyi" diyebiliriz.  
Ama bu fark **gerçek mi** yoksa **şans eseri mi?**

**Friedman Testi:** 3+ modeli karşılaştır, aralarında anlamlı fark var mı?  
Non-parametrik — normal dağılım varsaymaz (bizim verimiz için uygun).

**Wilcoxon + Bonferroni:** Hangi ikili çiftler arasında fark var?  
Bonferroni düzeltmesi → çoklu test yapınca yanlış pozitif riski artar, bunu düzeltir.

```
p < 0.05 → fark istatistiksel olarak anlamlı
p > 0.05 → fark şans eseri olabilir
```

---

## 🔍 Bonus Analizler

### Feature Stability (Özellik Kararlılığı)
20 farklı split'te MRMR hangi özellikleri seçti?  
Skor = 1.0 → her split'te seçildi = çok güvenilir sinyal  
Skor = 0.3 → sadece 6 split'te seçildi = güvenilmez

**Ne öğrettik?** Hangi klinik sinyalin gerçekten tutarlı olduğunu.

### Threshold Optimizasyonu (Youden J)
Varsayılan karar eşiği 0.5'tir. "Olasılık > 0.5 ise Papilödema de."  
Ama bu her zaman optimum değil.

**Youden J = Sensitivity + Specificity − 1**  
Bu değeri maksimize eden eşiği bulduk → genellikle 0.4-0.45 çıktı.  
Anlamı: biraz daha "şüpheci" ol, daha düşük olasılıkta bile Papilödema de.  
Tıbbi açıdan mantıklı: hastayı kaçırmak, yanlış alarm vermekten kötü.

### SHAP Analizi
"Model neden bu kararı verdi?" sorusuna cevap.  
Her özelliğin her tahmine katkısını ölçer.

```
Örnek Hasta #7:
  mean_intensity: +0.23  → papilödema yönünde itti
  variance:       +0.18  → papilödema yönünde itti  
  entropy:        -0.05  → normal yönünde itti
  ──────────────────────
  Net: +0.36 → Papilödema kararı
```

**Finans analojisi:** Portföy attribution — hangi hisse ne kadar kazandırdı/kaybettirdi?

---

## ✅ Sonuç — Ne Öğrendik?

| Konu | Öğrenilen Ders |
|------|----------------|
| Veri temizliği | PatientIndex çakışması gibi küçük hatalar büyük sorun çıkarır |
| Özellik seçimi | 746 özelliğin 736'sı gürültü olabilir — MRMR bu gürültüyü keser |
| Model seçimi | Tek model değil, ensemble → daha stabil |
| Parametre ayarı | Elle denemek yerine Optuna → akıllı arama |
| Değerlendirme | Accuracy yetmez, sensitivity ve AUC daha gerçekçi |
| Tekrarlanabilirlik | 20 split → tek şans eseri sonuç değil, gerçek performans |
| Yorumlanabilirlik | SHAP → "kara kutu" değil, açıklanabilir model |

---

## 📁 Proje Yapısı

```
papilodema_classification/
├── notebooks/
│   ├── 01_veri_kesfi.ipynb          → Veri tanıma, istatistikler
│   ├── 02_derin_eda.ipynb           → Detaylı görsel analiz
│   ├── 03_on_isleme.ipynb           → Temizlik, ölçekleme pipeline'ı
│   ├── 04_mrmr_ozellik_secimi.ipynb → 746 → 10 özellik
│   ├── 05_model_egitimi_optuna.ipynb→ RF, ET, GB + Optuna
│   ├── 06_ensemble_kalibrasyon.ipynb→ Soft Voting + kalibrasyon
│   ├── 07_istatistiksel_analiz.ipynb→ Friedman, Wilcoxon, Bonus D, E
│   └── 08_performans_grafikleri.ipynb→ 6 grafik + SHAP
├── figures/                         → Tüm grafikler (PNG)
├── results/                         → Metrik CSV'leri
├── report/                          → PDF ve Word rapor
└── data/                            → Ham veri
```

**GitHub:** https://github.com/barisaykent-spec/papilodema-classification

---

*Barış Refik Aykent · 244329058 · Makine Öğrenmesi Final Projesi · 2025*
