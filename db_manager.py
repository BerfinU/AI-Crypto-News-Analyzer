import sqlite3
import pickle
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__) 

class DBManager: 
    def __init__(self, db_path: str = "news.db"):
        self.db_path = db_path
        self.conn = None
        self.cur = None 
    
    def __enter__(self):
        """Context manager ile veritabanı bağlantısı aç."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
        self.init_db()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager ile bağlantıyı kapat."""
        if self.conn:
            if exc_type is None:
                self.conn.commit() # Hata yoksa değişiklikleri kaydet
            else:
                self.conn.rollback() # Hata varsa geri al
            self.conn.close()

    def init_db(self):
        """Veritabanı tablolarını ve indeksleri oluştur."""
        # Ana haberler tablosu
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE NOT NULL, author TEXT NOT NULL,
                title TEXT NOT NULL, content TEXT NOT NULL, tweet_time TEXT NOT NULL,
                tweet_id TEXT UNIQUE, scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                score REAL, importance_label TEXT, importance_confidence REAL,
                embedding BLOB, is_notified INTEGER DEFAULT 0, metrics TEXT,
                group_id INTEGER DEFAULT NULL,
                FOREIGN KEY (group_id) REFERENCES news_groups(group_id)
            )
        """)

        # Haber grupları tablosu
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS news_groups (
                group_id INTEGER PRIMARY KEY AUTOINCREMENT, main_news_id INTEGER NOT NULL UNIQUE,
                related_news_ids TEXT NOT NULL, llm_summary TEXT,
                last_processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (main_news_id) REFERENCES news(id)
            )
        """)

        # Performans için indeksler
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_tweet_time ON news(tweet_time DESC)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_importance_conf ON news(importance_confidence DESC)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_notified ON news(is_notified)")
        
        # İşlem logları tablosu
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tweets_processed INTEGER, new_tweets INTEGER, duplicates_found INTEGER,
                notifications_sent INTEGER, errors TEXT
            )
        """)
        self.conn.commit()
    
    def get_latest_tweet_id(self) -> Optional[str]:
        """En son çekilen tweet ID'sini getir."""
        self.cur.execute("SELECT tweet_id FROM news ORDER BY tweet_time DESC, id DESC LIMIT 1")
        row = self.cur.fetchone()
        return row["tweet_id"] if row else None

    def get_latest_tweet_id_for_account(self, author: str) -> Optional[str]:
        """Belirli bir hesap için en son tweet ID'sini getir."""
        self.cur.execute(
            "SELECT tweet_id FROM news WHERE author = ? AND tweet_id IS NOT NULL ORDER BY CAST(tweet_id AS INTEGER) DESC LIMIT 1",
            (author,)
        )
        row = self.cur.fetchone()
        return row["tweet_id"] if row else None

    def insert_news(self, tweet_data: Dict) -> int:
        """Yeni tweet verisini veritabanına ekle."""
        try:
            self.cur.execute("""
                INSERT OR IGNORE INTO news (
                    url, author, title, content, tweet_time, tweet_id,
                    score, importance_label, importance_confidence,
                    embedding, metrics, group_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """, (
                tweet_data["url"], tweet_data["author"], tweet_data["title"],
                tweet_data["content"], tweet_data["tweet_time"], tweet_data.get("tweet_id"),
                tweet_data.get("score"), tweet_data.get("importance_label"),
                tweet_data.get("importance_confidence"),
                pickle.dumps(tweet_data["embedding"]) if tweet_data.get("embedding") is not None else None,
                json.dumps(tweet_data.get("metrics", {}))
            ))
            row_id = self.cur.lastrowid or 0
            if row_id > 0:
                logger.info(f"INSERTED: ID={row_id}, Author={tweet_data['author']}, Title='{tweet_data['title'][:40]}...'")
            else:
                logger.info(f"IGNORED (duplicate?): Author={tweet_data['author']}, TweetID={tweet_data.get('tweet_id')}, Title='{tweet_data['title'][:40]}...'")
            return row_id
        except Exception as e:
            logger.error(f"Error inserting tweet: {e}")
            return 0

    def get_embeddings_for_dedup(self) -> List[Tuple[int, bytes]]:
        """Duplicate detection için mevcut embedding'leri getir."""
        self.cur.execute("SELECT id, embedding FROM news WHERE embedding IS NOT NULL")
        return [(row['id'], row['embedding']) for row in self.cur.fetchall()]

    def log_processing_run(self, stats: Dict):
        """İşlem istatistiklerini logla."""
        self.cur.execute("""
            INSERT INTO processing_log (tweets_processed, new_tweets, duplicates_found, errors)
            VALUES (?, ?, ?, ?)
        """, (
            stats.get('tweets_processed', 0), stats.get('new_tweets', 0),
            stats.get('duplicates_found', 0), json.dumps(stats.get('errors', []))
        ))

    def get_unclassified_tweets(self, limit: int = 50) -> List[Dict]:
        """Henüz sınıflandırılmamış tweetleri getir."""
        self.cur.execute("SELECT id, content FROM news WHERE importance_label IS NULL ORDER BY id ASC LIMIT ?", (limit,))
        return [dict(row) for row in self.cur.fetchall()]

    def update_classification(self, tweet_id: int, label: str, confidence: float, embedding_blob: bytes):
        """Tweet'in önem sınıflandırmasını güncelle."""
        self.cur.execute("UPDATE news SET importance_label = ?, importance_confidence = ?, embedding = ? WHERE id = ?",
                         (label, confidence, embedding_blob, tweet_id))

    def update_groups(self, groups_to_update: List[Dict], new_groups: List[Dict]):
        # Mevcut grupları güncelle
        for group in groups_to_update:
            self.cur.execute("UPDATE news_groups SET related_news_ids = ?, llm_summary = ?, last_processed_time = CURRENT_TIMESTAMP WHERE group_id = ?",
                             (json.dumps(group['related_news_ids']), group['llm_summary'], group['group_id']))
        # Yeni grupları oluştur
        for group in new_groups:
            self.cur.execute("INSERT INTO news_groups (main_news_id, related_news_ids, llm_summary) VALUES (?, ?, ?)",
                             (group['main_news_id'], json.dumps(group['related_news_ids']), group['llm_summary']))
            group_id = self.cur.lastrowid
            # İlgili haberleri gruba ata
            placeholders = ",".join("?" for _ in group['related_news_ids'])
            self.cur.execute(f"UPDATE news SET group_id = ? WHERE id IN ({placeholders})", [group_id] + group['related_news_ids'])

    def get_last_grouped_news_id(self) -> int:
        """Son gruplanmış haberin ID'sini getir."""
        self.cur.execute("SELECT MAX(id) as last_id FROM news WHERE group_id IS NOT NULL")
        row = self.cur.fetchone()
        return row['last_id'] if row and row['last_id'] is not None else 0

    def get_grouped_news_for_app(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Dashboard için gruplandırılmış haberleri getir."""
        time_limit = (datetime.now() - timedelta(hours=hours)).isoformat()
        query = f"""
            SELECT g.group_id, g.llm_summary, g.related_news_ids, n.id as main_news_id,
                   n.title, n.content, n.author, n.tweet_time, n.url,
                   n.importance_label, n.importance_confidence
            FROM news_groups g JOIN news n ON g.main_news_id = n.id
            WHERE n.tweet_time >= ?
            ORDER BY g.group_id DESC, n.importance_confidence DESC
            LIMIT ?
        """
        self.cur.execute(query, (time_limit, limit))
        groups = [dict(row) for row in self.cur.fetchall()]
        
        # Her grup için kaynak bilgilerini ekle
        for group in groups:
            related_ids = json.loads(group['related_news_ids'])
            placeholders = ",".join("?" for _ in related_ids)
            self.cur.execute(f"SELECT DISTINCT author, url FROM news WHERE id IN ({placeholders})", related_ids)
            sources_data = self.cur.fetchall()
            
            # Benzersiz kaynakları topla
            unique_sources = {}
            for row in sources_data:
                author = row['author']
                if author not in unique_sources:
                    unique_sources[author] = row['url']
            
            group['sources'] = list(unique_sources.keys())
            group['source_urls'] = unique_sources
        return groups

    def get_recent_ungrouped_news(self, last_id: int = 0, time_cutoff: str = None) -> List[Dict]:
        """Henüz gruplanmamış son haberleri getir."""
        if time_cutoff:
            self.cur.execute("""
                SELECT id, content, author, title, tweet_time FROM news
                WHERE id > ? AND group_id IS NULL 
                AND importance_label IS NOT NULL
                AND datetime(tweet_time) >= datetime(?)
                ORDER BY tweet_time DESC, id ASC
            """, (last_id, time_cutoff))
        else:
            self.cur.execute("""
                SELECT id, content, author, title, tweet_time FROM news
                WHERE id > ? AND group_id IS NULL AND importance_label IS NOT NULL
                ORDER BY tweet_time DESC, id ASC
            """, (last_id,))
        
        return [dict(row) for row in self.cur.fetchall()]

    def get_recent_groups(self, time_cutoff: str) -> List[Dict]:
        """Belirli tarihten sonraki grupları getir."""
        self.cur.execute("""
            SELECT g.group_id, g.main_news_id, g.related_news_ids, g.llm_summary 
            FROM news_groups g
            JOIN news n ON g.main_news_id = n.id
            WHERE datetime(n.tweet_time) >= datetime(?)
            ORDER BY g.group_id
        """, (time_cutoff,))
        return [dict(row) for row in self.cur.fetchall()]

    def get_high_importance_unnotified(self, threshold: float = 0.8) -> List[Dict]:
        """Yüksek önem derecesine sahip ve henüz bildirilmemiş tweetleri getir."""
        self.cur.execute("""
            SELECT id, url, author, title, content, tweet_time, 
                   importance_label, importance_confidence, metrics
            FROM news 
            WHERE is_notified = 0 
            AND importance_confidence >= ?
            AND importance_label IN ('Important', 'Very Important')
            ORDER BY importance_confidence DESC
            LIMIT 20
        """, (threshold,))
        return [dict(row) for row in self.cur.fetchall()]

    def mark_as_notified(self, tweet_ids: List[int]):
        """Belirtilen tweet ID'lerini bildirilmiş olarak işaretle."""
        placeholders = ",".join("?" for _ in tweet_ids)
        self.cur.execute(f"UPDATE news SET is_notified = 1 WHERE id IN ({placeholders})", tweet_ids)

    def get_stats(self, days: int = 1) -> Dict:
        """Son X günde işlenen tweet istatistiklerini getir."""
        # Toplam tweet sayısı
        self.cur.execute("""
            SELECT COUNT(*) as total_tweets
            FROM news 
            WHERE tweet_time >= datetime('now', ? || ' day')
        """, (f'-{days}',))
        total_tweets = self.cur.fetchone()['total_tweets']
        
        # Önem seviyesine göre dağılım
        self.cur.execute("""
            SELECT importance_label, COUNT(*) as count
            FROM news 
            WHERE tweet_time >= datetime('now', ? || ' day')
            AND importance_label IS NOT NULL
            GROUP BY importance_label
        """, (f'-{days}',))
        by_importance = {row['importance_label']: row['count'] for row in self.cur.fetchall()}
        
        # Yazar bazında dağılım
        self.cur.execute("""
            SELECT author, COUNT(*) as count
            FROM news 
            WHERE tweet_time >= datetime('now', ? || ' day')
            GROUP BY author
            ORDER BY count DESC
            LIMIT 10
        """, (f'-{days}',))
        by_author = {row['author']: row['count'] for row in self.cur.fetchall()}
        
        return {
            'total_tweets': total_tweets,
            'by_importance': by_importance,
            'by_author': by_author
        }