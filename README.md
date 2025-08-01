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
python -m venv venv
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

## Adım 2: Konfigürasyon

### 1. API Anahtarını Ayarlayın
Proje ana dizininde `.env` adında bir metin dosyası oluşturun. İçine, Google AI Studio'dan aldığınız Gemini API anahtarınızı aşağıdaki gibi yapıştırın:

```env
GOOGLE_API_KEY="AIzaSy...SİZİN_ANAHTARINIZ"
```

### 2. Genel Ayarları Yapılandırın
`config/` klasöründeki `config.yaml` dosyasını bir metin düzenleyici ile açın ve aşağıdaki alanları kendinize göre doldurun:

#### Model Ayarları:
- **`model -> classifier_path`**: Size verdiğim, eğitilmiş sınıflandırma modelini (`crypto_model_finetuned` klasörü) koyduğunuz yerin tam yolunu buraya yazmalısınız.

**Örnek:**
```yaml
model:
  classifier_path: "C:/Users/Berfin/Desktop/Proje/crypto_model_finetuned"
```

#### Telegram Ayarları:
```yaml
telegram:
  bot_token: "BOTFATHER_DAN_ALDIGINIZ_TOKEN"
  chat_id: "KANAL_VEYA_GRUP_ID_SI"  # Genellikle negatif bir sayıdır
```

## Adım 3: Çalıştırma

### 1. Veritabanını İlk Kez Oluşturun
Eğer projeyi ilk defa çalıştırıyorsanız, aşağıdaki komutu **sadece bir kez** çalıştırarak `news.db` dosyasını ve gerekli tabloları oluşturun:

```bash
python create_db.py
```

### 2. Ana Sistemi Başlatın (Scheduler)
Bu komut, veri çekme ve işleme döngüsünü başlatır. Bu terminali açık bırakmalısınız:

```bash
python scheduler.py
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

## Sorun Giderme

### Yaygın Sorunlar:

**Sanal ortam aktive edilmiyor:**
- Proje ana dizininde olduğunuzdan emin olun
- Python kurulumunu kontrol edin

**Playwright kurulum hatası:**
- Şunu deneyin: `pip install --upgrade pip`
- Sonra tekrar: `pip install playwright && playwright install`

**Model yolu hatası:**
- `crypto_model_finetuned` klasörünün var olduğundan emin olun
- `config/config.yaml` dosyasında tam yolu güncelleyin

**Dashboard yüklenmiyor:**
- Önce scheduler'ın çalıştığını kontrol edin
- Veritabanının `python create_db.py` ile oluşturulduğundan emin olun

## Proje Yapısı

```
ai-crypto-news-analyzer/
├── app.py                 # Streamlit dashboard
├── scheduler.py           # Ana otomasyon scripti
├── create_db.py          # Veritabanı başlatma
├── requirements.txt      # Python bağımlılıkları
├── .env                  # API anahtarları (oluşturun)
├── config/
│   └── config.yaml      # Konfigürasyon dosyası
└── crypto_model_finetuned/  # AI modeli (ekleyin)
```

## Ek Notlar

- İlk çalıştırmada sistem birkaç dakika sürebilir
- Scheduler çalışırken terminal penceresini kapatmayın
- Dashboard'da veriler gerçek zamanlı güncellenir
- Herhangi bir sorun yaşarsanız logları kontrol edin
