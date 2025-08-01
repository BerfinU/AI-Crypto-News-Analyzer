import schedule
import time
import logging
import signal
import sys
import yaml
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Transformers paralellik uyarılarını engelle
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

from x_news_scraper import CryptoTwitterScraper
from news_processor import ImportanceClassifier, BatchProcessor
from llm_grouper import NewsGrouper
from db_manager import DBManager
from telegram_notifier import run_telegram_notifications

# Logging ayarları
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Gereksiz kütüphane loglarını kapat
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('playwright').setLevel(logging.WARNING)

class CryptoNewsScheduler:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.db_path = self.config['database']['path']

        # Twitter scraper'ı başlat
        tw_cfg = self.config['twitter']
        self.scraper = CryptoTwitterScraper(accounts=tw_cfg['accounts'])
        
        # ML modeli yükle
        model_cfg = self.config['model']
        model_path = model_cfg['classifier_path']
        if not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            sys.exit(1)
        self.classifier = ImportanceClassifier(
            model_dir=model_path,
            embedding_model=model_cfg['embedding_model']
        )
        
        # Batch processor
        self.batch_processor = BatchProcessor(self.classifier, self.db_path)
        
        # LLM gruplamacı başlat
        self.llm_grouper = NewsGrouper(db_path=self.db_path)

        # Telegram ayarları
        self.bot_token = self.config['telegram'].get('bot_token', '')
        self.chat_id = self.config['telegram'].get('chat_id', '')
        self.notify_threshold = self.config['telegram']['notification_threshold']
        
        # İstatistik takibi
        self.stats = {
            'total_cycles': 0,
            'successful_scrapes': 0,
            'total_tweets_processed': 0,
            'telegram_notifications_sent': 0,
            'last_successful_scrape': None
        }
        
        # Graceful shutdown için sinyal yakalayıcılar
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Konfigürasyon dosyasını yükle, yoksa varsayılan oluştur."""
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            logger.info(f"Loaded config from {path}")
            return cfg
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            logger.info("Creating default config...")
            return self._create_default_config(path)

    def _create_default_config(self, path: str) -> Dict[str, Any]:
        """Varsayılan konfigürasyon dosyası oluştur."""        
        default_config = {
            'database': {'path': 'news.db'},
            'twitter': {
                'use_mock': False,
                'accounts': [
                    "CoinDesk", "TheBlock__", "Cointelegraph", "DecryptMedia",
                    "BitcoinMagazine", "WatcherGuru", "cryptonews", "coinbase"
                ]
            },
            'model': {
                'classifier_path': '/Users/berfinummetoglu/Desktop/crypto_news/crypto_model_finetuned',
                'embedding_model': 'all-MiniLM-L6-v2'
            },
            'schedule': {
                'scrape_interval_minutes': 5, 
                'scrape_limit': 15,
                'grouping_interval_minutes': 10,
                'daily_summary_time': '09:00'
            },
            'telegram': {
                'bot_token': '',
                'chat_id': '',
                'notification_threshold': 0.8
            }
        }
        # Dizin yoksa oluştur
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        logger.info(f"Created default config at {path}")
        return default_config
    
    def scrape_and_classify(self):
        """ADIM 1-2: Tweet çekme ve AI sınıflandırma."""
        logger.info("====== (1/3)NEWS GATHERING AND CLASSIFICATION BEGINS ======")
        cycle_start = time.time()
        
        try:
            self.stats['total_cycles'] += 1
            logger.info(f"Cycle #{self.stats['total_cycles']} starts...")
            
            # Twitter'dan son tweetleri çek
            logger.info("Fetching recent tweets...")
            tweets = self.scraper.fetch_recent_tweets(
                db_path=self.db_path,
                max_results=self.config['schedule']['scrape_limit']
            )
            
            if not tweets:
                logger.info("Yeni tweet bulunamadı.")
                # Sınıflandırılmamış tweetleri kontrol et
                stats = self.batch_processor.process_unclassified(batch_size=50)
                if stats['total_processed'] > 0:
                    logger.info(f"Eski tweetlerden {stats['total_processed']} sınıflandırıldı")
                elapsed = time.time() - cycle_start
                logger.info(f"Döngü süresi: {elapsed:.1f}s (Sonuç: Tweet yok)")
                return

            # Tweet istatistikleri  
            logger.info(f"{len(tweets)} === yeni tweet bulundu, sınıflandırılıyor...")
            authors = {}
            for tweet in tweets:
                authors[tweet['author']] = authors.get(tweet['author'], 0) + 1
            
            # Hesap bazında tweet dağılımını göster
            logger.info("Hesap bazında tweet sayıları:")
            for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  • {author}: {count} tweet")
            
            stats = self.classifier.process_tweets(tweets, self.db_path)
            
            self.stats['successful_scrapes'] += 1
            self.stats['total_tweets_processed'] += stats['tweets_processed']
            self.stats['last_successful_scrape'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            elapsed = time.time() - cycle_start
            logger.info(f"SINIFLANDİRMA TAMAMLANDI")
            logger.info(f"  İşlenen: {stats['tweets_processed']}")
            logger.info(f"  Yeni: {stats['new_tweets']}")
            logger.info(f"  Duplicate: {stats['duplicates_found']}")

            if stats.get('high_importance'):
                logger.info(" Yüksek Önemli Haberler:")
                for item in stats['high_importance'][:3]:
                    logger.info(f"  • {item['title'][:60]}... ({item['confidence']:.1%})")

        except Exception as e:
            logger.error(f"Haber çekme/sınıflandırma hatası: {e}", exc_info=True)
            elapsed = time.time() - cycle_start
            logger.info(f" Başarısız döngü süresi: {elapsed:.1f}s")

    def group_news_with_llm(self):
        """ADIM 3: LLM ile haber gruplama."""
        logger.info("====== (2/3) NEWS GROUPING WITH LLM BEGINS ======")
        start_time = time.time()
        
        try:
            self.llm_grouper.process_and_group_news()
            elapsed = time.time() - start_time
            logger.info(f" LLM gruplama tamamlandı ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f" LLM gruplama hatası ({elapsed:.1f}s): {e}", exc_info=True)

    def send_telegram_notifications(self):
        """ADIM 4: Telegram bildirimleri."""
        logger.info("====== (3/3) TELEGRAM NOTIFICATIONS ARE BEING SENT ======")
        
        # Telegram bot token ve chat ID kontrolü
        if not self.bot_token or not self.chat_id:
            logger.info(" Telegram bot token/chat ID tanımlanmamış, bildirimler atlanıyor")
            return
        
        try:
            notifications_sent = run_telegram_notifications(
                bot_token=self.bot_token,
                chat_id=self.chat_id,
                db_path=self.db_path,
                threshold=self.notify_threshold
            )
            
            self.stats['telegram_notifications_sent'] += notifications_sent
            
            if notifications_sent > 0:
                logger.info(f" {notifications_sent} Telegram bildirimi gönderildi")
            else:
                logger.info(" Gönderilecek yeni bildirim bulunamadı")
                
        except Exception as e:
            logger.error(f" Telegram bildirimi hatası: {e}", exc_info=True)

    def print_stats(self):
        """ Scheduler performans istatistiklerini yazdır."""
        logger.info(" SCHEDULER İSTATİSTİKLERİ:")
        logger.info(f" Toplam döngü: {self.stats['total_cycles']}")
        logger.info(f" Başarılı scrape: {self.stats['successful_scrapes']}")
        logger.info(f" Toplam tweet: {self.stats['total_tweets_processed']}")
        logger.info(f" Telegram bildirimi: {self.stats['telegram_notifications_sent']}")
        if self.stats['last_successful_scrape']:
            logger.info(f"    Son başarılı scrape: {self.stats['last_successful_scrape']}")

    def setup_schedule(self):
        sched = self.config['schedule']
        
        # Schedule ayarlarını yap - SADECE ana fonksiyonu zamanla
        schedule.every(sched['scrape_interval_minutes']).minutes.do(self.run_full_cycle)
        schedule.every(30).minutes.do(self.print_stats)
        
        logger.info(f" SCHEDULER YAPILANDI:")
        logger.info(f" Ana döngü (4 adım): Her {sched['scrape_interval_minutes']} dakikada")
        logger.info(f" İstatistik: Her 30 dakikada")

    def run_full_cycle(self):
        """4 adımı sırayla çalıştıran ana fonksiyon."""
        # ADIM 1-2: Tweet çekme ve sınıflandırma
        self.scrape_and_classify()
        time.sleep(5) 
        
        # ADIM 3: LLM gruplama
        self.group_news_with_llm()
        time.sleep(5)
        
        # ADIM 4: Telegram bildirimleri
        self.send_telegram_notifications()

    def run(self):
        logger.info(" CryptoNews Scheduler başlatılıyor...")
        logger.info(f" Takip edilen hesaplar ({len(self.config['twitter']['accounts'])}): {', '.join(self.config['twitter']['accounts'])}")
        
        # Telegram bildirimleri aktif mi kontrol et
        if self.bot_token and self.chat_id:
            logger.info(f" Telegram bildirimleri AKTIF (eşik: {self.notify_threshold})")
        else:
            logger.info(" Telegram bildirimleri DEVRE DIŞI")
        
        self.setup_schedule()
        
        logger.info(" BAŞLANGIÇ DÖNGÜSÜ çalıştırılıyor...")
        print("=" * 70)

        # İlk döngüyü hemen çalıştır
        self.run_full_cycle()
        
        logger.info(" Scheduler çalışıyor. Durdurmak için Ctrl+C.")
        logger.info(" Sonraki scrape: {} dakika sonra".format(self.config['schedule']['scrape_interval_minutes']))
        print("-" * 90)
        
        # Ana döngü - zamanlanmış görevleri çalıştır
        cycle_count = 0
        while True:
            schedule.run_pending()
            time.sleep(300)
            
            # Her 5 dakikada bir istatistikleri yazdır
            cycle_count += 1
            if cycle_count % 300 == 0:
                logger.info(" Scheduler aktif, sonraki görevler beklemede...")

    def _shutdown(self, signum, frame):
        logger.info(f" Sinyal {signum} alındı, kapatılıyor...")
        self.print_stats() 
        logger.info(" Scheduler kapatıldı")
        sys.exit(0)

if __name__ == "__main__":
    scheduler = CryptoNewsScheduler(config_path="config/config.yaml") 
    scheduler.run()