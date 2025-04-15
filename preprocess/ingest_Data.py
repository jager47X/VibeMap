import os
import logging
import pandas as pd
from pymongo import MongoClient, WriteConcern
from pymongo.errors import BulkWriteError
from config import MONGO_URI, DOCUMENT_PATH, DB_NAME,COLLECTION_NAME  # Ensure these are defined

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_csv_file():
    """Check if the CSV file exists and load it as a DataFrame."""
    if not os.path.exists(DOCUMENT_PATH):
        logger.error("File does not exist: %s", DOCUMENT_PATH)
        return None

    try:
        # Use 'latin1' for encoding flexibility
        df = pd.read_csv(DOCUMENT_PATH, encoding='latin1')
        required_cols = {"tweets_time", "username", "tweets"}
        if not required_cols.issubset(set(df.columns)):
            logger.error("CSV must contain columns: %s", required_cols)
            return None
        return df
    except Exception as e:
        logger.error("Error reading the CSV file: %s", e)
        return None

def ingest_csv_to_mongodb():
    """
    Ingest CSV tweets into MongoDB:
      - Connects to MongoDB with write concern w=0
      - Drops non-_id indexes
      - Loads CSV and removes NaN tweets
      - Filters out duplicates based on tweet text
      - Logs % progress during tweet processing
      - Performs bulk insertion
      - Recreates index on 'tweets'
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db.get_collection(COLLECTION_NAME , write_concern=WriteConcern(w=0))
        logger.info("Connected to MongoDB with write concern w=0.")

        # Drop all non-_id indexes
        existing_indexes = collection.index_information()
        for index in existing_indexes:
            if index != "_id":
                collection.drop_index(index)
                logger.info("Dropped index: %s", index)

        # Load CSV
        df = load_csv_file()
        if df is None or df.empty:
            logger.warning("No valid CSV data loaded.")
            return

        # Drop rows with missing tweets
        df = df.dropna(subset=["tweets"])
        logger.info("Valid tweet rows after dropping NaNs: %d", len(df))

        # Load existing tweets from DB
        existing_tweets = set(doc["tweets"] for doc in collection.find({}, {"tweets": 1}))
        logger.info("Loaded %d existing tweet texts from MongoDB.", len(existing_tweets))

        # Prepare documents with progress logging
        records = []
        skipped = 0
        total = len(df)

        for i, row in enumerate(df.itertuples(index=False), start=1):
            tweet_text = row.tweets
            if tweet_text in existing_tweets:
                skipped += 1
            else:
                records.append({
                    "tweets_time": row.tweets_time,
                    "username": row.username,
                    "tweets": tweet_text
                })

            if i % max(1, total // 20) == 0 or i == total:
                percent = (i / total) * 100
                logger.info("Processing progress: %.1f%% (%d/%d)", percent, i, total)

        if not records:
            logger.info("No new tweets to insert (all duplicates).")
            return

        # Insert new documents
        try:
            result = collection.insert_many(records, ordered=False)
            inserted = len(result.inserted_ids)
            logger.info("Inserted %d new tweet documents.", inserted)
        except BulkWriteError as bwe:
            inserted = bwe.details.get("nInserted", 0)
            logger.warning("Bulk insertion error. Inserted %d tweets.", inserted)

        # Recreate index on 'tweets'
        try:
            collection.create_index("tweets", unique=True)
            logger.info("Re-created unique index on 'tweets'.")
        except Exception as e:
            logger.warning("Could not create unique index on 'tweets': %s", e)

        logger.info("Tweet ingestion complete. Inserted: %d | Skipped: %d", inserted, skipped)
    except Exception as e:
        logger.error("Error during tweet ingestion: %s", e)
    finally:
        client.close()
        logger.info("MongoDB connection closed.")

if __name__ == '__main__':
    ingest_csv_to_mongodb()
