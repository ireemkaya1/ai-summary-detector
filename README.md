# 🧠 HUMAN_OR_AI

**HUMAN_OR_AI**, kullanıcının girdiği metnin **insan** mı yoksa **yapay zeka** tarafından mı üretildiğini tahmin eden bir web uygulamasıdır.

Proje, doğal dil işleme teknikleri ve farklı sınıflandırma algoritmalarını kullanarak Flask tabanlı bir arayüz üzerinden anlık tahminler sunar.

---

## 🎯 Proje Amacı

ChatGPT ve Gemini gibi yapay zeka araçlarının yaygınlaşmasıyla birlikte, üretilen içeriklerin kaynağını belirlemek giderek önem kazanmaktadır. Bu proje, akademik ve profesyonel ortamlarda metin doğrulama ihtiyacına çözüm sunmayı hedefler.

Amaç: Girilen bir metnin **İnsan** veya **Yapay Zeka** tarafından yazılıp yazılmadığını tespit etmek.

### Kapsam:

- Metin ön işleme ve analiz
- Çoklu makine öğrenmesi modelleri ile sınıflandırma
- Web arayüzü üzerinden gerçek zamanlı tahmin

---

## 🧠 Model ve Yöntem

### Kullanılan Modeller

- Logistic Regression
- Multinomial Naive Bayes
- SGD Classifier (Stochastic Gradient Descent)

### Metin Vektörizasyonu

- TF-IDF (Term Frequency – Inverse Document Frequency)
  - Kelime n-gramları (1-2)
  - Karakter n-gramları (3-5)

### Model Saklama

- Eğitilmiş modeller `.joblib` formatında kaydedilir
- Singleton Pattern ile optimize edilmiş model yükleme

---
## Ekran Görüntüleri

### 1. Ana Sayfa Arayüzü
Modern ve sade kullanıcı arayüzü.
![Ana Sayfa](images/home_page.png)

---

### 2. Tahmin Sonucu
Modelin analiz sonucu ve olasılık grafikleri.
![Tahmin Sonucu](images/prediction_result.png)