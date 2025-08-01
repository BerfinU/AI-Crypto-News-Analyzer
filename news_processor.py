import pickle
import json
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
from db_manager import DBManager
import os

logger = logging.getLogger(__name__) 

class ImportanceClassifier:
    """Haber önemini sınıflandır ve embedding ile duplicate kontrolü yap."""
    def __init__(self, model_dir: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_dir = model_dir
        # GPU varsa kullan, yoksa CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # BERT tabanlı önem sınıflandırıcısı yükle
        self.classifier = pipeline(  # zero-shot-classification
            "text-classification",
            model=model_dir,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Embedding modeli (benzerlik kontrolü için)
        self.embed_model = SentenceTransformer(embedding_model) 
        self.similarity_threshold = 0.85 # Benzerlik eşiği
    
    def classify_text(self, text: str) -> Tuple[str, float]: 
        """Metin önemini sınıflandır (Important/Medium/Unimportant)."""
        try:
            result = self.classifier(text, truncation=True, max_length=512)[0] 
            return result['label'], result['score']
        except Exception as e:
            logger.error(f"Error classifying text: {e}") 
            return "Medium", 0.5 # Hata durumunda varsayılan değer

    def get_embedding(self, text: str) -> np.ndarray: 
        """Metin için vektör temsili oluştur."""
        return self.embed_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    
    def is_duplicate(self, new_embedding: np.ndarray, existing_embeddings: List[Tuple[int, np.ndarray]]) -> Optional[int]:
        """Yeni embedding'in mevcut tweetlerle benzerliğini kontrol et."""
        if not existing_embeddings:
            return None
        
        new_emb_tensor = torch.tensor(new_embedding)
        
        # Her mevcut embedding ile karşılaştır
        for tweet_id, existing_emb in existing_embeddings:
            if not isinstance(existing_emb, np.ndarray):
                continue
            
            existing_tensor = torch.tensor(existing_emb)
            # Cosine similarity hesapla
            similarity = util.cos_sim(new_emb_tensor, existing_tensor).item()
            
            if similarity > self.similarity_threshold:
                logger.info(f"Similarity check: found similar to tweet {tweet_id} with score {similarity:.3f}")
                return tweet_id
        
        return None
    
    def process_tweets(self, tweets: List[Dict], db_path: str = "news.db") -> Dict: 
        """Tweet listesini sınıflandır ve veritabanına kaydet."""
        stats = {
            'tweets_processed': 0, 'new_tweets': 0, 'duplicates_found': 0,
            'high_importance': [], 'errors': []
        }
        
        with DBManager(db_path) as db: 
            # Mevcut embeddingleri yükle (duplicate kontrolü için)
            existing_embeddings = []
            for tweet_id, embedding_blob in db.get_embeddings_for_dedup():
                if embedding_blob:
                    try:
                        embedding = pickle.loads(embedding_blob)
                        existing_embeddings.append((tweet_id, embedding))
                    except Exception as e:
                        logger.error(f"Error loading embedding for tweet {tweet_id}: {e}")
            
            print(f" Processing {len(tweets)} tweets through AI classifier...")
            
            # Her tweet'i işle
            for i, tweet in enumerate(tweets, 1):
                try:
                    stats['tweets_processed'] += 1
                    
                    # İlerleme göstergesi
                    if i % 10 == 0 or i == len(tweets):
                        print(f"    Classifying tweet {i}/{len(tweets)}: {tweet['title'][:50]}...")
                    
                    # Tweet içeriği için embedding oluştur
                    embedding = self.get_embedding(tweet['content'])
                    
                    duplicate_of_id = self.is_duplicate(embedding, existing_embeddings)
                    if duplicate_of_id:
                        stats['duplicates_found'] += 1
                        print(f"    Duplicate found: '{tweet['title'][:50]}...' is a duplicate of existing tweet ID {duplicate_of_id}. Skipping.")
                        continue

                    # Önem sınıflandırması
                    label, confidence = self.classify_text(tweet['content'])
                
                    # Tweet verilerine ML sonuçlarını ekle
                    tweet['importance_label'] = label
                    tweet['importance_confidence'] = confidence
                    tweet['embedding'] = embedding
                    
                    # Veritabanına kaydet
                    row_id = db.insert_news(tweet)
                    if row_id:
                        stats['new_tweets'] += 1
                        existing_embeddings.append((row_id, embedding))
                        
                        # Yüksek önemli haberleri işaretle
                        if confidence >= 0.8 and label == 'Important':
                            print(f"   🔥 HIGH IMPORTANCE: {tweet['title'][:60]}... ({confidence:.1%})")
                            stats['high_importance'].append({
                                'id': row_id, 'title': tweet['title'], 'url': tweet['url'],
                                'label': label, 'confidence': confidence
                            })
                    
                except Exception as e:
                    error_msg = f"Error processing tweet {tweet.get('url', 'unknown')}: {e}"
                    logger.error(error_msg, exc_info=True)
                    stats['errors'].append(error_msg)

            # İşlem istatistiklerini logla
            db.log_processing_run(stats)
        return stats

class BatchProcessor:
    """Toplu olarak sınıflandırılmamış tweetleri işle."""
    def __init__(self, classifier: ImportanceClassifier, db_path: str = "news.db"):
        self.classifier = classifier
        self.db_path = db_path
    
    def process_unclassified(self, batch_size: int = 50) -> Dict:
        """Veritabanındaki sınıflandırılmamış tweetleri batch halinde işle."""
        stats = {'total_processed': 0, 'batches': 0, 'errors': []}
        
        with DBManager(self.db_path) as db:
            while True:
                # Sınıflandırılmamış tweetleri getir
                unclassified = db.get_unclassified_tweets(limit=batch_size)
                if not unclassified:
                    logger.info("No unclassified tweets found to process.")
                    break
                
                stats['batches'] += 1
                print(f" Processing batch #{stats['batches']} with {len(unclassified)} unclassified tweets.")
                
                # Her tweet'i sınıflandır
                for tweet in unclassified:
                    try:
                        # ML sınıflandırması
                        label, confidence = self.classifier.classify_text(tweet['content'])
                        embedding = self.classifier.get_embedding(tweet['content'])
                        
                        # Veritabanında güncelle
                        db.update_classification(
                            tweet['id'], label, confidence, pickle.dumps(embedding)
                        )
                        stats['total_processed'] += 1
                        
                        # Yüksek güvenilirlikli sonuçları göster
                        if confidence >= 0.8:
                            print(f"    {label} ({confidence:.1%}): {tweet['content'][:50]}...")
                            
                    except Exception as e:
                        error_msg = f"Batch processing error for tweet {tweet['id']}: {e}"
                        logger.error(error_msg, exc_info=True)
                        stats['errors'].append(error_msg)
                
                print(f"    Batch #{stats['batches']} completed")
        
        if stats['total_processed'] > 0:
            logger.info(f"Batch processing complete. Processed {stats['total_processed']} tweets in {stats['batches']} batches.")
        return stats