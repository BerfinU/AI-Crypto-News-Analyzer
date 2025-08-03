import logging
from typing import List, Dict
import asyncio
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from db_manager import DBManager
import time
import re

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        # Rate limiting ayarları
        self.max_messages_per_minute = 20
        self.message_times = [] # Son gönderilen mesaj zamanları
    
    def escape_markdown_v2(self, text: str) -> str:
        """
        Telegram MarkdownV2 formatı için özel karakterleri escape et.
        Telegram'ın sıkı markdown kuralları nedeniyle gerekli.
        """
        if not text:
            return ""
        
        # Telegram MarkdownV2'de escape edilmesi gereken karakterler
        special_chars = [
            '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', 
            '|', '{', '}', '.', '!'
        ]
        
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        
        return text
    
    async def _send_message(self, text: str, parse_mode: str = None):
        """Send a single message with rate limiting."""
        current_time = time.time()
        # Son 1 dakika içindeki mesajları filtrele
        self.message_times = [t for t in self.message_times if current_time - t < 60]
        
        # Rate limit kontrolü
        if len(self.message_times) >= self.max_messages_per_minute:
            wait_time = 60 - (current_time - self.message_times[0])
            logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        try:
            # Ana mesaj gönderme denemesi
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=False
            )
            self.message_times.append(time.time())
            return True
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            # Markdown hatası varsa plain text ile tekrar dene
            if "can't parse entities" in str(e).lower() or "must be escaped" in str(e).lower():
                try:
                    logger.info("Retrying message without markdown formatting...")
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        parse_mode=None,  # Plain text
                        disable_web_page_preview=False
                    )
                    self.message_times.append(time.time())
                    return True
                except TelegramError as e2:
                    logger.error(f"Telegram error on retry: {e2}")
            return False
    
    def format_tweet_notification(self, tweet: Dict) -> str:
        """Tweet'i Telegram bildirimi için formatla."""
        emoji_map = {
            'Important': '🔥',
            'Medium': '📊',
            'Unimportant': '💬'
        }
        
        emoji = emoji_map.get(tweet.get('importance_label', ''), '📰')
        importance = tweet.get('importance_label', 'Unknown').upper()
        
        # Metin içeriklerini markdown-safe hale getir
        title = self.escape_markdown_v2(tweet.get('title', 'No Title'))
        content = tweet.get('content', '')
        if len(content) > 400:
            content = content[:397] + "..."
        content = self.escape_markdown_v2(content)
        author = self.escape_markdown_v2(tweet.get('author', 'Unknown'))
        
        url = tweet.get('url', '#')
        
        # Mesaj formatını oluştur
        message = f"{emoji} *{self.escape_markdown_v2(importance)}*\n\n"
        message += f"*{title}*\n\n"
        message += f"{content}\n\n"
        message += f"👤 {author}\n"
        message += f"🔗 [View Tweet]({url})\n"
        
        # Varsa engagement metrikleri ekle
        if tweet.get('metrics'):
            import json
            try:
                metrics = json.loads(tweet['metrics']) if isinstance(tweet['metrics'], str) else tweet['metrics']
                if metrics:
                    likes = metrics.get('likes', 0)
                    retweets = metrics.get('retweets', 0)
                    message += f"\n❤️ {likes} \\| 🔄 {retweets}"
            except:
                pass
        
        return message
    
    async def notify_high_importance_tweets(self, db_path: str = "news.db", threshold: float = 0.8):
        """Yüksek önemli tweetler için bildirim gönder."""
        notifications_sent = 0
        
        with DBManager(db_path) as db:
            # Henüz bildirim gönderilmemiş yüksek önemli tweetleri getir
            tweets = db.get_high_importance_unnotified(threshold=threshold)
            
            if not tweets:
                logger.info("No new high importance tweets to notify")
                return notifications_sent
            
            logger.info(f"Found {len(tweets)} high importance tweets to notify")
            
            notified_ids = []
            # Her tweet için bildirim gönder
            for tweet in tweets:
                message = self.format_tweet_notification(tweet)
                
                success = await self._send_message(message, ParseMode.MARKDOWN_V2)
                
                if success:
                    notified_ids.append(tweet['id'])
                    notifications_sent += 1
                    logger.info(f"Notified tweet {tweet['id']}: {tweet['title'][:50]}...")
                else:
                    logger.error(f"Failed to notify tweet {tweet['id']}")
            
            # Başarılı bildirimleri veritabanında işaretle
            if notified_ids:
                db.mark_as_notified(notified_ids)
        
        return notifications_sent
    
    async def send_daily_summary(self, db_path: str = "news.db"):
        """Günlük crypto haber özeti gönder."""
        with DBManager(db_path) as db:
            # Son 24 saatin istatistiklerini al
            stats = db.get_stats(days=1)
            
            # Özet mesajını oluştur
            message = " DAILY CRYPTO NEWS SUMMARY\n\n"
            message += f" Total tweets: {stats['total_tweets']}\n\n"
            
            # Önem seviyesine göre dağılım
            if stats['by_importance']:
                message += "By Importance:\n"
                for label, count in sorted(stats['by_importance'].items()):
                    message += f"• {label}: {count}\n"
                message += "\n"
            
            # En aktif kaynaklar (top 5)
            if stats['by_author']:
                message += "Top Sources:\n"
                for author, count in list(stats['by_author'].items())[:5]:
                    message += f"• {author}: {count} tweets\n"
            
            # Plain text olarak gönder (istatistikler için markdown gereksiz)
            await self._send_message(message, parse_mode=None)


def run_telegram_notifications(bot_token: str, chat_id: str, db_path: str = "news.db", threshold: float = 0.8):
    """
    Senkron wrapper - scheduler'dan çağırılabilmesi için.
    Asyncio event loop oluşturur ve bildirimleri çalıştırır.
    """
    notifier = TelegramNotifier(bot_token, chat_id)
    
    # Yeni event loop oluştur (scheduler thread-safe olması için)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Asenkron bildirim fonksiyonunu çalıştır
        notifications_sent = loop.run_until_complete(
            notifier.notify_high_importance_tweets(db_path, threshold)
        )
        logger.info(f"Sent {notifications_sent} notifications")
        return notifications_sent
    finally:
        # Event loop'u temizle
        loop.close()


if __name__ == "__main__":
    import logging
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("====== (3/3) TELEGRAM NOTIFICATIONS ARE BEING SENT ======")
    
    # Scheduler'daki ayarlarla aynı
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
    
    if not bot_token or not chat_id:
        logger.info(" Telegram bot token/chat ID tanımlanmamış, bildirimler atlanıyor")
        exit()
    
    try:
        notifications_sent = run_telegram_notifications(
            bot_token=bot_token,
            chat_id=chat_id,
            db_path="news.db",
            threshold=0.8
        )
        
        if notifications_sent > 0:
            logger.info(f" {notifications_sent} Telegram bildirimi gönderildi")
        else:
            logger.info(" Gönderilecek yeni bildirim bulunamadı")
            
    except Exception as e:
        logger.error(f" Telegram bildirimi hatası: {e}")
