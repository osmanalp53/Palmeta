
# 🪷 Palmeta – El Çizgilerinden Üretken Yorumlama Sistemi

**Palmeta**, bir el görüntüsünden elde edilen çizgi yoğunluklarını analiz ederek, kişisel ve sembolik yorumlar üreten üretken yapay zeka projesidir. Bu sistem, görselden elde edilen sayısal verileri bir karakter tabanlı **Transformer modeli** ile işleyerek kullanıcıya anlamlı bir hayat metni sunar.

---

## 🔍 Nasıl Çalışır?

1. Kullanıcıdan bir **el görseli** (JPG/PNG) alınır.
2. Görselden 8 farklı anatomik bölgeye ait **yoğunluk vektörü** çıkarılır:
   - Venüs, Jüpiter, Satürn, Apollo, Merkür, Luna, Mars, Kalp Merkezi
3. Bu vektör, koşullu giriş olarak **Transformer modeline** verilir.
4. Her bölge için farklı bir prompt ile anlamlı ve yaratıcı metin üretilir.
5. Sonuçlar birleştirilerek kullanıcıya sunulur.

---

## 🗂️ Dosya Yapısı

```bash
Palmeta/
├── data/
│   ├── hand_images/          # Kullanıcıdan alınan ham el görselleri
│   └── processed_vectors/    # Görsellerden çıkarılan yoğunluk vektörleri
├── models/
│   └── transformer_generator.py  # Transformer modeli
├── src/
│   ├── region_features.py        # Görsel öznitelik çıkarımı
│   ├── generate_text.py          # Metin üretimi
│   ├── predict_from_image.py     # Uçtan uca tahmin sistemi
├── char_transformer.pth          # Eğitilmiş model ağırlıkları
└── README.md
```

---

## 💬 Örnek Çıktı

```text
🔮 Venüs bölgesi duygular, romantizm ve sezgilerle ilgilidir.
Senin Venüs bölgen şöyle: Silik duygularını bastırıyor olabilirsin.

🔥 Mars bölgesi, içsel enerjiyi ve mücadele gücünü temsil eder.
Senin Mars bölgen oldukça belirgin, hırsın seni ileri taşıyor.
```

---

## 🧠 Kullanılan Veri Seti

Bu projede el görselleri, **Kaggle** üzerindeki açık veri setinden alınmıştır:  
📎 [Hands and Palm Images Dataset](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset)

> Not: Görselleri `data/hand_images/` klasörüne yerleştirmeniz yeterlidir.

---

## ⚙️ Gereksinimler

Tüm bağımlılıklar aşağıdaki komutla kurulabilir:

```bash
pip install -r requirements.txt
```

---
## 🛠️ Python Dosyalarının Görevleri

| Dosya Adı | Görev Açıklaması |
|-----------|------------------|
| `extract_dataset_features.py` | Görsellerden öznitelik çıkarımı |
| `region_features.py` | Elin bölgesel analizini yapar |
| `train_region_model.py` | Görsel öznitelik çıkarıcı modelin eğitimi |
| `train_transformer.py` | Hayat hikayesi üreten transformer modelin eğitimi |
| `generate_text.py` | Eğitilen transformer ile yazı üretimi yapar |
| `predict_from_image.py` | El görselinden doğrudan hayat hikayesi üretimi |
 
## 👤 Geliştirici

**Osman Alp Polatoğlu**  
📘 Yapay Zeka Mühendisliği – 3. Sınıf  
🎓 Öğrenci No: 220212005

---

## 🚀 Notlar

- Proje tamamen sıfırdan geliştirilen bir üretken yapay zeka yapısıdır.
- Açıklamalar ve prompt'lar kültürel ve sembolik anlamlara göre düzenlenmiştir.
- Bu sistemde sınıflandırıcı değil, **koşullu üretici model** kullanılmıştır.
