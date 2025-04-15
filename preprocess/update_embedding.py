import json
import logging
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MONGO_URI, COLLECTION, DB_NAME

# Global constant for max tokens.
MAX_TOTAL_TOKENS = 8000
EMBEDDING_TARGET = "tweets"  # The field in the document where we make embedding from.

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load efficient embedding model (GPU-friendly)
logger.info("Loading embedding model: intfloat/e5-small-v2")
model = SentenceTransformer("intfloat/e5-small-v2")
logger.info("Model loaded.")

def update_corpus_embeddings(config):
    logger.info("Connecting to the database...")
    with MongoClient(MONGO_URI) as client:
        db = client[DB_NAME]
        embedding_collection = db[config["embedding_collection_name"]]
        logger.info("Connected to the database.")
        total_count = embedding_collection.count_documents({})

        count_missing = embedding_collection.count_documents({"embedding": {"$exists": False}})
        logger.info(f"Total documents in collection: {total_count}")
        logger.info(f"Documents missing embeddings: {count_missing}")

        logger.info("Choose processing mode:")
        logger.info("  [c] Continue processing missing embeddings only")
        logger.info("  [b] Start from beginning (process all documents)")
        mode = input("Enter your choice (c/b): ").strip().lower()

        if mode not in ['c', 'b']:
            logger.warning("Invalid input. Defaulting to 'continue' mode.")
            mode = 'c'

        if mode == 'c':
            docs = list(embedding_collection.find({"embedding": {"$exists": False}}))
            processed = total_count - count_missing
        else:
            docs = list(embedding_collection.find({}))
            result = embedding_collection.update_many({}, {"$unset": {"embedding": ""}})
            logger.info("Removed existing embeddings from %d documents.", result.modified_count)
            processed = 0

        logger.info("Starting embedding update...")
        user_input = input("Proceed with embedding update? (y/n): ").strip().lower()
        if user_input != 'y':
            logger.warning("Skipping embedding update as per user input.")
            return

        def process_single_document(doc):
            try:
                text = doc.get(EMBEDDING_TARGET, "").strip()
                if not text:
                    return False
                input_text = "query: " + text
                embedding = model.encode(input_text, normalize_embeddings=True)
                unique_field = config.get("unique_index", "tweets_time")
                with MongoClient(MONGO_URI) as local_client:
                    db_local = local_client[DB_NAME]
                    collection_local = db_local[config["embedding_collection_name"]]
                    collection_local.update_one(
                        {unique_field: doc[unique_field]},
                        {"$set": {"embedding": embedding.tolist()}}
                    )
                return True
            except Exception as e:
                logger.error("Error processing document with id %s: %s", doc.get(config.get("unique_index", "tweets_time")), str(e))
                return False
        import multiprocessing as mp
        num_workers = mp.cpu_count() - 1  # Leave one core free
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_document, doc) for doc in docs]
            for i, future in enumerate(as_completed(futures)):
                processed += 1 if future.result() else 0
                percentage = (processed / total_count) * 100 if total_count > 0 else 100
                logger.info(f"Progress: {processed}/{total_count} ({percentage:.2f}%)")

        logger.info("Successfully completed embedding updates in the database.")

if __name__ == "__main__":
    logger.info("Available configurations:")
    for i, config in enumerate(COLLECTION, start=1):
        doc_type = config.get("document_type", "Unknown")
        logger.info("%d: %s", i, doc_type)

    try:
        selected_num = int(input("Enter configuration number: ").strip())
        if selected_num < 1 or selected_num > len(COLLECTION):
            raise ValueError("Selection out of range")
    except Exception as e:
        logger.warning("Invalid configuration number provided. Defaulting to 1.")
        selected_num = 1

    config = COLLECTION[selected_num - 1]
    logger.info("Using configuration: %s", config.get("document_type", "Unknown"))
    logger.info("Selected configuration details: %s", json.dumps(config, indent=4))
    update_corpus_embeddings(config)