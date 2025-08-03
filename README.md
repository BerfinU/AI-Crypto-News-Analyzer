# Kurulum ve Ayar Rehberi

Bu rehber, projeyi kendi bilgisayarınızda sıfırdan kurup çalıştırmak için gereken tüm adımları içerir.

## Gereksinimler

- **Python** (versiyon 3.9 veya üstü önerilir)
- **pip** (Python paket yöneticisi)

## Adım 1: Kurulum

### 1. Proje Dosyalarını Alın
Proje dosyalarını bilgisayarınızda bir klasöre indirin/kopyalayın.

```bash
git clone https://github.com/BerfinU/ai-crypto-news-analyzer.git
cd ai-crypto-news-analyzer
```

### 2. Sanal Ortam Oluşturun
Bu, projenin kütüphanelerini sisteminizdeki diğer projelerden izole tutar. Terminali proje ana dizininde açın ve çalıştırın:

```bash
python3 -m venv venv
```

### 3. Sanal Ortamı Aktive Edin

**Windows'ta:**
```bash
venv\Scripts\activate
```

**macOS/Linux'ta:**
```bash
source venv/bin/activate
```

### 4. Gerekli Paketleri Yükleyin
Projenin ana uygulamasını çalıştırmak için gereken tüm kütüphaneler bu dosyada listelenmiştir:

```bash
pip install -r requirements.txt
```

### 5. Playwright Tarayıcılarını İndirin
Bu adım, veri toplayıcının (scraper) çalışabilmesi için **zorunludur**:

```bash
pip install playwright
playwright install
```

## Adım 2: AI Modelini İndirin

### Hazır Eğitilmiş Model:
Projenin çalışması için eğitilmiş BERT tabanlı sınıflandırma modeli gereklidir:

**Model İndirme Linki:** [Google Drive'dan İndirin](https://drive.google.com/file/d/1bgpQUT6FihQwgmBDWhPdcKqlLh5Az8mz/view?usp=drive_link)

### Model Kurulum Adımları:

#### Manuel İndirme:
1. **Yukarıdaki linke tıklayın**
2. **"crypto_model_finetuned.zip"** dosyasını indirin
3. **Proje ana klasörüne çıkarın**
4. **Klasör yapısı şöyle olmalı:**
   ```
   ai-crypto-news-analyzer/
   ├── crypto_model_finetuned/
   │   ├── pytorch_model.bin
   │   ├── config.json
   │   ├── tokenizer.json
   │   └── diğer model dosyaları...
   ├── app.py
   └── diğer proje dosyaları...
   ```

#### Terminal ile İndirme (İsteğe Bağlı):
```bash
# Model dosyasını direkt indirin
curl -L "https://drive.google.com/file/d/1bgpQUT6FihQwgmBDWhPdcKqlLh5Az8mz/view?usp=drive_link" -o crypto_model_finetuned.zip

# Çıkarın
unzip crypto_model_finetuned.zip

# ZIP dosyasını silin (isteğe bağlı)
rm crypto_model_finetuned.zip
```

### Model Hakkında:
- **Model Türü:** BERT-based Fine-tuned Classifier
- **Sınıflar:** Important, Medium, Unimportant
- **Dil:** İngilizce kripto haberleri
- **Dosya Boyutu:** ~450MB
- **Accuracy:** %80+

## Adım 3: Model Eğitimi (İsteğe Bağlı)

### Kendi Modelinizi Eğitmek İsterseniz:
Projede `crypto_news_model_finetuning.ipynb` dosyası bulunmaktadır. Bu notebook ile kendi AI modelinizi eğitebilirsiniz:

1. **Google Colab'a Yükleyin**: Notebook dosyasını Google Colab'a yükleyin
2. **Veri Setinizi Hazırlayın**: Kripto haberleri içeren CSV dosyanızı hazırlayın
3. **Notebook'u Çalıştırın**: Adım adım hücreleri çalıştırarak modelinizi eğitin
4. **Modeli İndirin**: Eğitim tamamlandığında modeli bilgisayarınıza indirin

**Not:** Model eğitimi birkaç saat sürebilir ve güçlü bir GPU gerektirir. Bu nedenle Google Colab önerilir.

## Adım 4: Konfigürasyon

### 1. API Anahtarını Ayarlayın
Proje ana dizininde `.env` adında bir metin dosyası oluşturun. İçine, Google AI Studio'dan aldığınız Gemini API anahtarınızı aşağıdaki gibi yapıştırın:

```env
GOOGLE_API_KEY="AIzaSy...SİZİN_ANAHTARINIZ"
```

### 2. Genel Ayarları Yapılandırın
`config/` klasöründeki `config.yaml` dosyasını bir metin düzenleyici ile açın ve aşağıdaki alanları kendinize göre doldurun:

#### Model Ayarları:
- **`model -> classifier_path`**: İndirdiğiniz model klasörünün tam yolunu buraya yazın.

**Örnek:**
```yaml
model:
  classifier_path: "/Users/berfin/Desktop/ai-crypto-news-analyzer/crypto_model_finetuned"
  embedding_model: "all-MiniLM-L6-v2"
```

#### Telegram Ayarları:
```yaml
TELEGRAM_BOT_TOKEN=BOTFATHER_DAN_ALDIGINIZ_TOKEN
TELEGRAM_CHAT_ID=KANAL_VEYA_GRUP_ID
```

## Adım 4: Çalıştırma

### 1. Veritabanını İlk Kez Oluşturun
Eğer projeyi ilk defa çalıştırıyorsanız, aşağıdaki komutu **sadece bir kez** çalıştırarak `news.db` dosyasını ve gerekli tabloları oluşturun:

```bash
python3 create_db.py
```

### 2. Ana Sistemi Başlatın (Scheduler)
Bu komut, veri çekme ve işleme döngüsünü başlatır. Bu terminali açık bırakmalısınız:

```bash
python3 scheduler.py
```

Ekranda logların akmaya başladığını göreceksiniz. Bu, sistemin çalıştığı anlamına gelir.

### 3. Dashboard'u Görüntüleyin
- **Yeni bir** terminal penceresi açın
- Aynı şekilde sanal ortamı tekrar aktive edin (`venv\Scripts\activate`)
- Streamlit arayüzünü başlatmak için aşağıdaki komutu girin:

```bash
streamlit run app.py
```

Otomatik olarak tarayıcınızda `http://localhost:8501` adresinde bir sekme açılacak ve dashboard'u görebileceksiniz.

## Hızlı Başlangıç Komutları

İlk kurulumdan sonra, projeyi çalıştırmak için sadece şu komutları kullanmanız yeterli:

```bash
# Terminal 1 - Arka Plan Sistemi
source venv/bin/activate  # Windows'ta: venv\Scripts\activate
python scheduler.py

# Terminal 2 - Dashboard  
source venv/bin/activate  # Windows'ta: venv\Scripts\activate
streamlit run app.py
```


## Proje Yapısı

```
ai-crypto-news-analyzer/
├── app.py                 # Streamlit dashboard
├── scheduler.py           # Ana otomasyon scripti
├── create_db.py          # Veritabanı başlatma
├── crypto_news_model_finetuning.ipynb  # Model eğitim notebook'u
├── requirements.txt      # Python bağımlılıkları
├── .env                  # API anahtarları (oluşturun)
├── config/
│   └── config.yaml      # Konfigürasyon dosyası
└── crypto_model_finetuned/  # AI modeli (eğitin veya alın)
```

## Model Eğitimi Detayları

### Notebook İçeriği (`crypto_news_model_finetuning.ipynb`):
- **Veri Ön İşleme**: Kripto haber metinlerinin temizlenmesi
- **BERT Fine-tuning**: Transformer tabanlı model eğitimi
- **Sınıflandırma**: Important/Medium/Unimportant kategorileri
- **Model Değerlendirme**: Accuracy, F1-score metrikleri
- **Model Export**: Eğitilmiş modelin kaydedilmesi

### Eğitim Süreci:
1. Veri setinizi hazırlayın (CSV formatında)
2. Google Colab'da notebook'u açın
3. GPU runtime'ı etkinleştirin
4. Hücreleri sırayla çalıştırın
5. Eğitim tamamlandığında modeli indirin
6. Model klasörünü proje dizinine yerleştirin

## Ek Notlar

- İlk çalıştırmada sistem birkaç dakika sürebilir
- Scheduler çalışırken terminal penceresini kapatmayın
- Dashboard'da veriler gerçek zamanlı güncellenir
- Herhangi bir sorun yaşarsanız logları kontrol edin
