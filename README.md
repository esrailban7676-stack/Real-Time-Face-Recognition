# 👁️ Real-Time Biometric Face Recognition & Verification System
### 🚀 Gerçek Zamanlı Biyometrik Yüz Tanıma ve Güvenlik Doğrulama Sistemi

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Field-Computer_Vision-FF6F00?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

<p align="center">
  <b>Canlı kamera akışı üzerinden önişlem adımları uygulayarak biyometrik yüz tespiti ve kimlik doğrulama gerçekleştiren yapay zeka/görüntü işleme projesi.</b>
</p>

</div>

---

## 📌 Proje Mimarisi ve Akış Şeması (Pipeline)

Sistem ham kamera görüntüsünü alır, çok aşamalı görüntü işleme süzgecinden geçirir ve anlık biyometrik karar üretir:

```text
┌──────────────────────┐
│  Canlı Kamera Akışı  │ (Webcam Video Stream)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Yüz Tespiti (ROI)   │ (Haar Cascade Classifier)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  ÖNİŞLEM ADIMLARI    │ ➔ 1. Gri Tonlama (Grayscale)
│  (Preprocessing)     │ ➔ 2. Boyut Standardizasyonu (100x100)
└──────────┬───────────┘ ➔ 3. Histogram Eşitleme (CLAHE Işık Dengesi)
           ▼
┌──────────────────────┐
│  Özellik Karşılaştırma│ ➔ Şablon Korelasyonu (Normalized Cross-Correlation)
│  (Feature Matching)  │ ➔ Histogram Benzerlik Analizi (Correlation Distance)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   KARAR MEKANİZMASI  │ ➔ Eşik Değer Üstü: 🟢 YEŞİL ("SENSİN")
│   (Threshold Engine) │ ➔ Eşik Değer Altı: 🔴 KIRMIZI ("SEN DEĞİLSİN")
└──────────────────────┘
