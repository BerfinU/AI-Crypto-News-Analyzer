import os
import json
import logging
from typing import List, Dict
from db_manager import DBManager
from datetime import datetime, timedelta

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class NewsGrouper:
    def __init__(self, db_path: str = "news.db"):
        self.db_path = db_path
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _create_prompt(self, input_data: Dict) -> str:
        """Create the prompt with improved instructions"""
        
        base_prompt = """
        TASK: You are an expert news analyst grouping RECENT cryptocurrency news articles.

        STRICT RULES:
        1. Only group news that are about the SAME SPECIFIC EVENT or STORY
        2. Only group news from the SAME TIME PERIOD (within 2-3 days of each other)
        3. Don't group generic topics like "Bitcoin price" unless they're about the same specific price movement
        4. Don't group old news with new news
        5. When in doubt, keep items separate

        INPUT FORMAT:
        - "existing_groups": Previously created news groups
        - "new_items": New ungrouped articles to process

        GROUPING CRITERIA:
        GOOD TO GROUP:
        - Multiple reports about Trump signing GENIUS Act
        - Different outlets covering the same company announcement
        - Various perspectives on the same regulatory decision

        DON'T GROUP:
        - Generic Bitcoin price mentions from different days
        - Old news with new news (check dates!)
        - Different topics that just use similar words
        - Generic market updates unless about same specific event

        OUTPUT FORMAT:
        Return ONLY valid JSON:
        {
          "updated_groups": [
            {
              "group_id": 1,
              "related_news_ids": [101, 105, 123],
              "llm_summary": "Specific event summary with timeframe"
            }
          ],
          "new_groups": [
            {
              "main_news_id": 125,
              "related_news_ids": [125, 128],
              "llm_summary": "Summary of this specific news event"
            }
          ]
        }
        
        ---
        DATA TO PROCESS:
        """

        return base_prompt + json.dumps(input_data, indent=2) 

    def _call_llm(self, prompt: str) -> Dict:  
        """Send request to LLM and parse response"""
        try:
            response = self.model.generate_content(prompt) 
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            logger.info(f"LLM response received (length: {len(cleaned_response)} chars)")
            return json.loads(cleaned_response)
        except Exception as e:
            logger.error(f"LLM API call or JSON parsing error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                raw_response = response.text[:500]
                logger.error(f"Raw response: {raw_response}")
            return {"updated_groups": [], "new_groups": []}

    def process_and_group_news(self):
        """Main method to process and group news items"""
        logger.info("Starting improved news grouping process...")
        
        with DBManager(self.db_path) as db:
            last_id = db.get_last_grouped_news_id()
            
            # Sadece son 7 gündeki haberleri al
            recent_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            new_items = db.get_recent_ungrouped_news(last_id, recent_cutoff)
            
            if not new_items:
                logger.info("No recent news items to group.")
                return

            logger.info(f"Found {len(new_items)} recent items to be grouped.")

            # Sadece son 7 gündeki grupları al
            existing_groups = db.get_recent_groups(recent_cutoff)
            logger.info(f"Found {len(existing_groups)} recent groups.")

            input_data = {
                "existing_groups": existing_groups,
                "new_items": new_items
            }
            
            prompt = self._create_prompt(input_data)
            logger.info(f"Sending improved prompt to LLM (length: {len(prompt)} chars)")
            
            llm_result = self._call_llm(prompt)
            
            if not llm_result or ('updated_groups' not in llm_result and 'new_groups' not in llm_result):
                logger.error("Could not get valid grouping result from LLM.")
                return

            updated_count = len(llm_result.get("updated_groups", []))
            new_count = len(llm_result.get("new_groups", []))

            logger.info(f"LLM returned {updated_count} updated groups and {new_count} new groups")

            db.update_groups( 
                llm_result.get("updated_groups", []),
                llm_result.get("new_groups", [])
            )
            
            logger.info(f"Improved grouping complete. {updated_count} groups updated, {new_count} new groups created.")


if __name__ == "__main__":
    import logging
    import time
    
    # Scheduler ile aynı logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("====== (2/3) NEWS GROUPING WITH LLM BEGINS ======")
    start_time = time.time()
    
    try:
        grouper = NewsGrouper(db_path="news.db")
        grouper.process_and_group_news()
        elapsed = time.time() - start_time
        logger.info(f" LLM gruplama tamamlandı ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f" LLM gruplama hatası ({elapsed:.1f}s): {e}", exc_info=True)
