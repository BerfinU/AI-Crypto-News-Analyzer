import os
import certifi
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from db_manager import DBManager 

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

logger = logging.getLogger(__name__)

class CryptoTwitterScraper:
    """X (Twitter) üzerinden kripto haberlerini çeker."""
    def __init__(self, bearer_token: str = None, accounts: List[str] = None):
        self.accounts = accounts or [
            "CoinDesk", "TheBlock__", "Cointelegraph", "DecryptMedia",
            "BitcoinMagazine", "WatcherGuru", "cryptonews", "coinbase"   
        ]
        self.base_url = "https://twitter.com/{}"
        self.problem_accounts = {}

    def parse_iso(self, ts: str) -> datetime:
        """ISO formatındaki string'i timezone-aware datetime objesine çevirir."""
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception as e:
            logger.warning(f"Could not parse timestamp {ts}: {e}")
            return datetime.now(timezone.utc)

    def extract_tweet_id_from_url(self, url: str) -> str:
        """Tweet URL'inden tweet ID'sini çıkarır."""
        try:
            return url.split('/status/')[-1].split('?')[0].split('/')[0] # URL'den ID'yi alır
        except Exception as e:
            logger.debug(f"Could not extract tweet ID from {url}: {e}")
            return None

    def scrape_account(self, handle: str, limit: int, since_id: str = None, is_initial: bool = False) -> List[Dict]:
        """Tek bir hesaptan haber çeker - Geliştirilmiş hata yönetimi ve kararlılık."""
        tweets = []
        url = self.base_url.format(handle)
        
        if handle in self.problem_accounts:
            failure_count = self.problem_accounts[handle]
            if failure_count >= 5: 
                logger.warning(f"Skipping {handle} due to repeated failures ({failure_count})")
                return []
        
        if is_initial:
            logger.info(f"INITIAL SCRAPE: {handle}, fetching {limit} tweets")
        else:
            logger.info(f"DELTA SCRAPE: {handle}, since_id={since_id}")

        with sync_playwright() as p:
            browser = None
            context = None
            page = None
            
            try:
                browser = p.firefox.launch(
                    headless=True,
                    args=[
                        '--no-sandbox', 
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--no-first-run',
                        '--disable-background-timer-throttling',
                        '--disable-renderer-backgrounding'
                    ]
                )
                
                # Context ayarları
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={'width': 1366, 'height': 768},
                    locale='en-US',
                    timezone_id='America/New_York'
                )
                # Yeni bir sayfa aç
                page = context.new_page()
                
                max_retries = 3
                page_loaded = False
                
                for retry in range(max_retries):
                    try:
                        logger.info(f"Navigating to {url} (attempt {retry + 1}/{max_retries})")
                        
                        # Önce sayfaya git
                        response = page.goto(url, timeout=30000, wait_until='domcontentloaded')
                        
                        if response and response.status >= 400:
                            logger.warning(f"HTTP {response.status} for {handle}")
                            if retry == max_retries - 1:
                                raise Exception(f"HTTP {response.status}")
                            time.sleep(10)
                            continue
                        
                        # Tweet'lerin yüklenmesini bekle
                        logger.info(f"Waiting for tweet articles to load for {handle}...")
                        page.wait_for_selector('article[data-testid="tweet"]', timeout=25000)
                        
                        # Sayfanın tamamen yüklendiğinden emin ol
                        time.sleep(3)
                        
                        # En az bir tweet var mı kontrol et
                        tweet_count = page.locator('article[data-testid="tweet"]').count()
                        if tweet_count == 0:
                            raise Exception("No tweets found on page")
                            
                        logger.info(f"Articles loaded for {handle}. Found {tweet_count} tweet elements. Starting scrape.")
                        page_loaded = True
                        break
                        
                    except PlaywrightTimeoutError as e:
                        logger.warning(f"Timeout loading {handle} (attempt {retry + 1}): {e}")
                        if retry < max_retries - 1:
                            time.sleep(5 * (retry + 1))  # Exponential backoff
                    except Exception as e:
                        logger.warning(f"Error loading {handle} (attempt {retry + 1}): {e}")
                        if retry < max_retries - 1:
                            time.sleep(5)
                # Eğer sayfa hala yüklenmediyse, problemi kaydet
                if not page_loaded:
                    self.problem_accounts[handle] = self.problem_accounts.get(handle, 0) + 1
                    logger.error(f"Could not load page for {handle} after {max_retries} attempts")
                    return []

                collected = 0 # Toplam tweet sayısı
                seen_tweet_ids = set() 
                max_scrolls = 12 if is_initial else 8 
                scrolls = 0 # Kaç kez scroll yapıldı
                consecutive_old_tweets = 0 # Kaç tane eski tweet bulundu
                no_new_content_count = 0 # İçerik olmayan tweet sayısı
                stale_page_count = 0 # Sayfa boşsa bu sayılır

                while collected < limit and scrolls < max_scrolls:
                    try:
                        # Tweet elementlerini al
                        articles = page.locator('article[data-testid="tweet"]').all()
                        
                        if not articles or len(articles) == 0:
                            logger.warning(f"No articles found on page for {handle} at scroll {scrolls}")
                            stale_page_count += 1
                            if stale_page_count >= 3:
                                break
                            # Sayfayı yenile
                            page.reload(timeout=20000)
                            time.sleep(3)
                            continue
                        
                        new_tweets_in_scroll = 0 
                        stale_page_count = 0
                        
                        logger.debug(f"Processing {len(articles)} articles for {handle} (scroll {scrolls + 1})") 
                        
                        for i, article in enumerate(articles): 
                            if collected >= limit:
                                break

                            try:
                                href = None
                                try:
                                    # Strateji 1: Direkt link bulma
                                    link_elements = article.locator('a[href*="/status/"]').all()
                                    for link_elem in link_elements:
                                        href_candidate = link_elem.get_attribute('href', timeout=3000)
                                        if href_candidate and '/status/' in href_candidate and len(href_candidate.split('/status/')[-1]) > 10:
                                            href = href_candidate
                                            break
                                    
                                    # Strateji 2: Time element içindeki link
                                    if not href:
                                        time_elem = article.locator('time').first
                                        if time_elem.count() > 0:
                                            parent_link = time_elem.locator('xpath=./ancestor::a[contains(@href, "/status/")]').first
                                            if parent_link.count() > 0:
                                                href = parent_link.get_attribute('href', timeout=2000)
                                
                                except Exception as e:
                                    logger.debug(f"Link extraction failed for article {i}: {e}")
                                    continue
                                
                                if not href: 
                                    continue

                                tweet_url = f"https://x.com{href}" if href.startswith('/') else href
                                tweet_id = self.extract_tweet_id_from_url(tweet_url)
                                
                                if not tweet_id or tweet_id in seen_tweet_ids: 
                                    continue
                                    
                                seen_tweet_ids.add(tweet_id) # Tweet ID'lerini takip et
                                
                                try:
                                    article_text = article.inner_text(timeout=3000) # Tweet içeriğini al
                                    if any(pin_word in article_text[:200].lower() for pin_word in ['pinned', 'sabitlenmiş', 'pinned tweet']):
                                        logger.debug(f"Skipping pinned tweet from {handle}")
                                        continue # Pinned tweet'leri atla
                                except:
                                    pass
                                
                                timestamp_str = datetime.now(timezone.utc).isoformat() # Varsayılan olarak güncel zaman
                                try:
                                    time_selectors = ['time[datetime]', 'time', '[datetime]'] 
                                    for selector in time_selectors: 
                                        try:
                                            time_elem = article.locator(selector).first
                                            if time_elem.count() > 0:
                                                datetime_attr = time_elem.get_attribute('datetime', timeout=2000)
                                                if datetime_attr:
                                                    timestamp_str = datetime_attr
                                                    break
                                        except:
                                            continue
                                except Exception as e:
                                    logger.debug(f"Timestamp extraction failed: {e}")

                                if collected == 0 and is_initial:
                                    try:
                                        tweet_dt = self.parse_iso(timestamp_str) 
                                        max_age_days = 45 
                                        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)

                                        if tweet_dt < cutoff_date:
                                            logger.warning(
                                                f"SKIPPING ACCOUNT: First tweet for '{handle}' is too old ({tweet_dt.date()}). "
                                                f"Will retry in next cycle."
                                            )
                                            self.problem_accounts[handle] = self.problem_accounts.get(handle, 0) + 0.3
                                            return []
                                    except Exception as e:
                                        logger.debug(f"Age check failed for {handle}: {e}")

                                if not is_initial and since_id: # Eğer delta scrape ise, since_id kontrolü yap
                                    try:
                                        logger.debug(f"[since_id kontrolü] since_id: {since_id}, tweet_id: {tweet_id}")
                                        try:
                                            current_id = int(tweet_id)
                                            since_id_int = int(since_id)
                                            if current_id <= since_id_int:
                                                consecutive_old_tweets += 1
                                                logger.debug(f"Old tweet found (int): {current_id} <= {since_id_int} (consecutive: {consecutive_old_tweets})")
                                                if consecutive_old_tweets >= 7:
                                                    logger.info(f"Found {consecutive_old_tweets} consecutive old tweets for {handle}, stopping")
                                                    collected = limit
                                                    break
                                                continue
                                            else:
                                                consecutive_old_tweets = 0
                                        except Exception as e:
                                            logger.debug(f"Int karşılaştırma hatası: {e}, string olarak denenecek.")
                                            if tweet_id <= since_id:
                                                consecutive_old_tweets += 1
                                                logger.debug(f"Old tweet found (str): {tweet_id} <= {since_id} (consecutive: {consecutive_old_tweets})")
                                                if consecutive_old_tweets >= 7:
                                                    logger.info(f"Found {consecutive_old_tweets} consecutive old tweets for {handle}, stopping (string compare)")
                                                    collected = limit 
                                                    break
                                                continue
                                            else:
                                                consecutive_old_tweets = 0
                                    except Exception as e:
                                        logger.debug(f"Could not compare tweet IDs for {handle}: {tweet_id} vs {since_id} - {e}")
                                        continue
                                
                                content = None 
                                try:
                                    # İçerik için farklı seçiciler dene
                                    content_selectors = [ 
                                        '[data-testid="tweetText"]',
                                        '[data-testid="tweetText"] span',
                                        'div[dir="auto"] span',
                                        '[lang] span',
                                        'div[data-testid="tweetText"]'
                                    ]
                                    
                                    for selector in content_selectors: 
                                        try:
                                            content_elems = article.locator(selector).all()
                                            if content_elems:
                                                content_parts = []
                                                for elem in content_elems[:5]: 
                                                    text = elem.inner_text(timeout=2000)
                                                    if text and text.strip():
                                                        content_parts.append(text.strip())
                                                
                                                if content_parts:
                                                    content = ' '.join(content_parts)
                                                    break
                                        except:
                                            continue
                                    
                                    if not content:
                                        full_text = article.inner_text(timeout=3000)
                                        lines = full_text.split('\n')
                                        clean_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('@') and 'Replying to' not in line]
                                        if clean_lines:
                                            content = clean_lines[0]
                                            
                                except Exception as e:
                                    logger.debug(f"Content extraction failed for {handle}: {e}")

                                if not content or len(content.strip()) < 15: # İçerik çok kısa ise atla
                                    no_new_content_count += 1
                                    logger.debug(f"No valid content found for tweet {tweet_id} from {handle}")
                                    continue

                                if len(content) > 1000:
                                    content = content[:500] + "..."
                                
                                no_new_content_count = 0 
                                
                                title = content.split('\n')[0][:200].strip()
                                if not title:
                                    title = content[:200].strip()

                                # Tweet verisini oluştur
                                tweet_data = {
                                    'url': tweet_url, 
                                    'tweet_id': tweet_id, 
                                    'title': title,
                                    'content': content.strip(), 
                                    'author': handle,
                                    'tweet_time': timestamp_str, 
                                    'metrics': {}
                                }
                                
                                tweets.append(tweet_data) # Tweet verisini listeye ekle
                                collected += 1
                                new_tweets_in_scroll += 1
                                
                                logger.debug(f"COLLECTED TWEET {collected}/{limit} FROM {handle}: {title[:50]}...") 
                                
                            except Exception as e: 
                                logger.debug(f"Error processing article {i} for {handle}: {e}")
                                continue

                        if collected >= limit: 
                            break
                        
                        if no_new_content_count > 15:
                            logger.warning(f"Too many articles without content for {handle}, stopping")
                            break
                        
                        scrolls += 1
                        logger.debug(f"Scroll {scrolls}/{max_scrolls} for {handle}, found {new_tweets_in_scroll} new tweets")
                        
                        if new_tweets_in_scroll == 0 and scrolls > 3: 
                            consecutive_empty_scrolls = consecutive_empty_scrolls + 1 if 'consecutive_empty_scrolls' in locals() else 1
                            if consecutive_empty_scrolls >= 3:
                                logger.info(f"No new tweets found in last 3 scrolls for {handle}, stopping")
                                break
                        else:
                            consecutive_empty_scrolls = 0
                        
                        try:
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            time.sleep(4)
                            
                            page.wait_for_timeout(1000)
                        except Exception as e:
                            logger.debug(f"Scroll error for {handle}: {e}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error in scroll loop for {handle}: {e}")
                        break
                    
            except Exception as e:
                logger.error(f"Critical error scraping {handle}: {e}", exc_info=True)
                self.problem_accounts[handle] = self.problem_accounts.get(handle, 0) + 1
                
            finally:
                try:
                    if page: page.close()
                    if context: context.close()
                    if browser: browser.close()
                except:
                    pass

        # Eğer tweet sayısı azsa, problem hesabı olarak işaretle
        if tweets and handle in self.problem_accounts:
            self.problem_accounts[handle] = max(0, self.problem_accounts[handle] - 0.5)
            if self.problem_accounts[handle] <= 0:
                del self.problem_accounts[handle]

        logger.info(f"Scraped {len(tweets)} tweets from {handle}")
        return tweets

    def fetch_recent_tweets(self, db_path: str, max_results: int = 10) -> List[Dict]:
        """Ana tweet çekme fonksiyonu."""
        all_tweets = []
        logger.info("Checking for new tweets on a per-account basis...")
        
        successful_accounts = 0
        total_accounts = len(self.accounts)

        # Eğer veritabanı yoksa, yeni bir tane oluştur
        with DBManager(db_path) as db: 
            for i, account in enumerate(self.accounts, 1): 
                try:
                    logger.info(f"PROCESSING ACCOUNT {i}/{total_accounts}: {account}") 
                    
                    since_id = db.get_latest_tweet_id_for_account(account) 
                    logger.info(f"[since_id LOG] Account: {account}, since_id: {since_id}") 
                    is_initial_for_account = since_id is None 
                    
                    if is_initial_for_account: 
                        logger.info(f" INITIAL SCRAPE for '{account}'. Fetching up to {max_results} tweets.")
                        limit = max_results
                    else:
                        logger.info(f" DELTA SCRAPE for '{account}'. Fetching new tweets since id: {since_id}")
                        limit = 25  # Makul limit
                    
                    tweets = self.scrape_account(
                        handle=account, 
                        limit=limit, 
                        since_id=since_id,
                        is_initial=is_initial_for_account
                    )
                    
                    if tweets:
                        all_tweets.extend(tweets)
                        successful_accounts += 1
                        logger.info(f" ✅ SUCCESS: {len(tweets)} TWEETS FROM {account}")
                        print("-" * 20)
                    else:
                        logger.warning(f" ❌ NO TWEETS FOUND FOR {account}")
                        print("-" * 20)
                    
                    time.sleep(2 + (i % 3))
                    
                except Exception as e:
                    logger.error(f"Error processing {account}: {e}", exc_info=True)
                    continue
        
        logger.info(f" SCRAPE SUMMARY: {successful_accounts}/{total_accounts} accounts successful")
        
        if self.problem_accounts:
            logger.info(f"Problem accounts: {dict(self.problem_accounts)}")
        
        if not all_tweets:
            logger.info(" No new tweets found across all accounts.")
            return []
            
        all_tweets.sort(key=lambda x: x['tweet_time'], reverse=True)
        
        # Duplicate tweet kontrolü - URL bazında da kontrol et
        unique_tweets = []
        seen_ids = set()
        seen_urls = set()
        
        for tweet in all_tweets:
            tweet_id = tweet['tweet_id']
            tweet_url = tweet['url']
            # Eğer hem ID hem de URL daha önce görülmemişse, ekle
            if tweet_id not in seen_ids and tweet_url not in seen_urls: 
                unique_tweets.append(tweet)
                seen_ids.add(tweet_id)
                seen_urls.add(tweet_url)
        
        # Eğer duplicate tweet varsa, logla
        removed_count = len(all_tweets) - len(unique_tweets)
        if removed_count > 0:
            logger.info(f"🔄 Removed {removed_count} duplicate tweets")
            
        logger.info(f"✅ SCRAPE COMPLETE: Found {len(unique_tweets)} unique tweets from {successful_accounts} accounts")
        return unique_tweets


class MockTwitterScraper:
    """Test için mock scraper"""
    def __init__(self, bearer_token: str = None, accounts: List[str] = None): # Mock scraper, gerçek Twitter API'si yerine kullanılacak
        self.accounts = accounts or ["TestAccount"] # Test için kullanılacak hesap
    
    def fetch_recent_tweets(self, db_path: str = None, max_results: int = 10) -> List[Dict]:
        import random
        mock_tweets = [] 
        for i in range(min(3, max_results)):
            
            mock_tweets.append({
                'url': f'https://twitter.com/TestAccount/status/{1000000 + i}',
                'tweet_id': str(1000000 + i), 
                'title': f'Test crypto news {i+1}',
                'content': f'This is test content for crypto news number {i+1}. Bitcoin price analysis and market trends.',
                'author': 'TestAccount', 
                'tweet_time': datetime.now(timezone.utc).isoformat(),
                'metrics': {'likes': random.randint(10, 100), 'retweets': random.randint(5, 50)}
            })
        return mock_tweets


    import logging
    import time
    from db_manager import DBManager
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("====== TWEET ÇEKME VE KAYDETME ======")
    
    # CryptoTwitterScraper bu dosyada tanımlı
    scraper = CryptoTwitterScraper()
    tweets = scraper.fetch_recent_tweets(db_path="news.db", max_results=15)
    
    if not tweets:
        logger.info("❌ Hiç tweet çekilemedi!")
        exit()
    
    # Tweet'leri RAW olarak veritabanına kaydet
    logger.info(f"📱 {len(tweets)} tweet çekildi, veritabanına kaydediliyor...")
    
    with DBManager("news.db") as db:
        saved_count = 0
        for tweet in tweets:
            # AI sınıflandırması olmadan kaydet
            tweet['importance_label'] = None
            tweet['importance_confidence'] = None  
            tweet['embedding'] = None
            
            row_id = db.insert_news(tweet)
            if row_id > 0:
                saved_count += 1
    
    logger.info(f"✅ {saved_count} tweet veritabanına kaydedildi!")
