import sqlite3
def init_db(db_path: str = "news.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Veritabanını ve tabloları oluştur
    cur.execute("DROP TABLE IF EXISTS news")
    cur.execute("DROP TABLE IF EXISTS processing_log")
    cur.execute("DROP TABLE IF EXISTS news_groups")
#
    cur.execute("""
    CREATE TABLE news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            author TEXT,
            title TEXT,
            content TEXT,
            tweet_time TEXT,
            tweet_id TEXT UNIQUE,
            scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            score REAL DEFAULT NULL,
            importance_label TEXT DEFAULT NULL,
            importance_confidence REAL DEFAULT NULL,
            embedding BLOB DEFAULT NULL,
            is_notified INTEGER DEFAULT 0,
            metrics TEXT,
            group_id INTEGER DEFAULT NULL, -- YENİ: Hangi gruba ait olduğunu belirtir
            FOREIGN KEY (group_id) REFERENCES news_groups(group_id) -- YENİ
    )
    """)
    # Haber işleme loglarını tutacak tablo
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tweets_processed INTEGER,
            new_tweets INTEGER,
            duplicates_found INTEGER,
            notifications_sent INTEGER,
            errors TEXT
        )
    """)
    # Haber gruplarını tutacak tablo
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_groups (
            group_id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_news_id INTEGER NOT NULL UNIQUE, -- Grupta ana haber olarak gösterilecek haberin ID'si
            related_news_ids TEXT NOT NULL, -- Bu gruba dahil olan tüm haber ID'lerinin JSON listesi
            llm_summary TEXT, -- LLM tarafından oluşturulan grubun özeti
            last_processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (main_news_id) REFERENCES news(id)
        )
    """)

    conn.commit()
    conn.close()
    print("Database ve tablolar başarıyla oluşturuldu/güncellendi.")

if __name__ == "__main__":
    init_db()