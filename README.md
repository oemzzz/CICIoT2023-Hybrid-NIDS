# CICIoT2023 Hybrid NIDS

IoT ağlarındaki siber saldırıları tespit etmek için geliştirdiğim iki aşamalı hibrit bir Saldırı Tespit Sistemi (NIDS) projesi. 

Bilecik Şeyh Edebali Üniversitesi Bilgisayar Mühendisliği bitirme çalışması olarak hazırladığım bu projede, derin öğrenmenin özellik çıkarma gücünü makine öğrenmesinin sınıflandırma yeteneğiyle birleştirdim.

---

## Proje Ne Yapıyor?
* **Veri Seti:** CICIoT2023 veri setinin dengelenmiş %20'lik alt kümesi (34 farklı sınıf).
* **Mimari:** 
  1. **1. Aşama:** Ham ağ verisinden 128 boyutlu derin özellikler çıkaran çok ölçekli bir **1D-CNN**.
  2. **2. Aşama:** Çıkarılan bu özelliklerle ham veriyi birleştirip ($46 + 128 = 174$ boyut) **XGBoost**, **Random Forest** ve **LightGBM** modellerine besleme.
* **Optimizasyon:** Sınıf dengesizliğini çözmek için eğitim setine özel örnek ağırlıklandırması (`sample_weight`) uygulandı.

---

## Özet Sonuçlar (Macro-Average)

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Base)** | 0.993 | 0.803 | 0.805 | 0.800 |
| **XGBoost (Hybrid)** | 0.993 | 0.791 | 0.803 | 0.794 |
| **Random Forest (Base)** | 0.992 | 0.891 | 0.751 | 0.779 |
| **Random Forest (Hybrid)** | 0.990 | 0.893 | 0.732 | 0.762 |
| **LightGBM (Base)** | 0.992 | 0.764 | 0.742 | 0.745 |
| **LightGBM (Hybrid)** | 0.993 | 0.780 | 0.742 | 0.751 |
| **Ensemble (Majority Vote)** | **0.993** | **0.843** | **0.761** | **0.784** |

---

##  Çalıştırma

1. Projeyi klonla:
   ```bash
   git clone [https://github.com/oemzzz/CICIoT2023-Hybrid-NIDS.git](https://github.com/oemzzz/CICIoT2023-Hybrid-NIDS.git)
   cd CICIoT2023-Hybrid-NIDS
