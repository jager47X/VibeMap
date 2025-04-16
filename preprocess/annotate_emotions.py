import os
import sys
import logging
from pymongo import MongoClient, ASCENDING

# ensure config.py is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from config import MONGO_URI, DB_NAME, COLLECTION_NAME, LABEL_COLLECTION, EMOTION_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def annotate_emotions():
    """
    Loop through each document in COLLECTION_NAME in ascending _id order,
    skip any already labeled in LABEL_COLLECTION,
    prompt for an emotion label (using EMOTION_LABELS),
    and upsert into LABEL_COLLECTION.
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    source = db[COLLECTION_NAME]
    target = db[LABEL_COLLECTION]

    try:
        cursor = source.find().sort("_id", ASCENDING)
        for doc in cursor:
            doc_id = doc["_id"]
            # skip if already labeled
            if target.count_documents({"_id": doc_id}, limit=1):
                continue

            tweet_text = doc.get("tweets", "<no tweet text>")
            print("\n" + "-" * 60)
            print(f"ID:    {doc_id}")
            print(f"Tweet: {tweet_text}\n")

            # Show available labels
            print("Available labels:")
            for idx, name in EMOTION_LABELS.items():
                print(f"  {idx}: {name}")
            print()

            raw = input("Enter label number (blank=skip, q=quit): ").strip()
            if raw.lower() == "q":
                logger.info("User aborted annotation.")
                break
            if not raw:
                logger.info(f"Skipped {doc_id}")
                continue

            # Validate and map to label
            try:
                label_idx = int(raw)
                label_name = EMOTION_LABELS[label_idx]
            except (ValueError, KeyError):
                logger.warning(f"Invalid label '{raw}' — skipping {doc_id}")
                continue

            # Upsert the label into target collection
            target.update_one(
                {"_id": doc_id},
                {"$set": {"label_idx": label_idx, "label": label_name}},
                upsert=True
            )
            logger.info(f"Labeled {doc_id!r} → {label_idx} ({label_name!r})")
    finally:
        client.close()
        logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    annotate_emotions()