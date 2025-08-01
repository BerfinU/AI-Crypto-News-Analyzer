from datetime import datetime
from db_manager import DBManager

def view_all_chronological():
    with DBManager() as db:
        db.cur.execute("""
            SELECT id, tweet_id, author, title, content, tweet_time, 
                   importance_label, importance_confidence, url, scraped_time
            FROM news 
            ORDER BY id ASC
        """)
        
        tweets = db.cur.fetchall()
        
        print(f"TÜM TWEET'LER")
        print(f"Toplam: {len(tweets)} tweet")
        print("="*120)
        
        for i, tweet in enumerate(tweets, 1):
            try:
                if tweet['tweet_time']:
                    tweet_dt = datetime.fromisoformat(tweet['tweet_time'].replace('Z', '+00:00'))
                    tweet_time_str = tweet_dt.strftime('%d/%m/%Y %H:%M')
                else:
                    tweet_time_str = "Bilinmiyor"
            except:
                tweet_time_str = tweet['tweet_time']
            
            try:
                if tweet['scraped_time']:
                    scraped_dt = datetime.fromisoformat(tweet['scraped_time'])
                    scraped_time_str = scraped_dt.strftime('%d/%m/%Y %H:%M:%S')
                else:
                    scraped_time_str = "Bilinmiyor"
            except:
                scraped_time_str = tweet['scraped_time']
            
            importance = tweet['importance_label'] or "Beklemede"
            confidence = f"{tweet['importance_confidence']:.1%}" if tweet['importance_confidence'] else "N/A"
            
            print(f"#{i:3d} | DB ID: {tweet['id']:3d} | Tweet ID: {tweet['tweet_id']}")
            print(f"      Yazar: {tweet['author']:15s} | Tweet Zamanı: {tweet_time_str}")
            print(f"      Önem: {importance:12s} ({confidence:5s}) | Çekilme: {scraped_time_str}")
            print(f"      Başlık: {tweet['title']}")
            print(f"      İçerik: {tweet['content'][:100]}...")
            print(f"      URL: {tweet['url']}")
            print("-" * 120)

if __name__ == "__main__":
    view_all_chronological()