import logging
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from config import MONGO_URI, DB_NAME, EMOTIONAL_LEVEL_COLLECTION,EMOTION_LABELS

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def ingest_emotional_levels_with_embedding():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[EMOTIONAL_LEVEL_COLLECTION]
        logger.info("Connected to MongoDB DB: '%s', collection: '%s'", DB_NAME, EMOTIONAL_LEVEL_COLLECTION)

        # Clear any existing documents in the emotional level collection
        delete_result = collection.delete_many({})
        logger.info("Cleared %d existing emotional level mapping documents.", delete_result.deleted_count)

        # Load SentenceTransformer model to compute embeddings for emotion labels
        logger.info("Loading embedding model: intfloat/e5-small-v2")
        model = SentenceTransformer("intfloat/e5-small-v2")
        logger.info("Model loaded.")

        # Prepare mapping records with computed embeddings
        records = []
        for cluster, label in EMOTION_LABELS.items():
            # Compute the embedding for the emotion label
            embedding = model.encode(label, normalize_embeddings=True)
            record = {
                "cluster": cluster,
                "emotion_level": label,
                "embedding": embedding.tolist()
            }
            records.append(record)

        # Insert the mapping documents into MongoDB
        result = collection.insert_many(records)
        logger.info("Inserted %d emotional level mapping documents.", len(result.inserted_ids))

        # Create a unique index on 'cluster'
        collection.create_index("cluster", unique=True)
        logger.info("Created unique index on 'cluster'.")

        # Create a unique index on 'emotion_level'
        collection.create_index("emotion_level", unique=True)
        logger.info("Created unique index on 'emotion_level'.")

    except Exception as e:
        logger.error("Error during emotional level ingestion: %s", e)
    finally:
        client.close()
        logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    ingest_emotional_levels_with_embedding()
    