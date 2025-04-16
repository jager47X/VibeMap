import logging
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from config import MONGO_URI, DB_NAME, EMOTIONAL_LEVEL_COLLECTION,EXTENDED_EMOTION_LABELS,EMOTION_LABELS



# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ingest_emotional_levels_grouped():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[EMOTIONAL_LEVEL_COLLECTION]
        logger.info("Connected to DB: '%s', collection: '%s'", DB_NAME, EMOTIONAL_LEVEL_COLLECTION)

        # Clear existing entries
        deleted = collection.delete_many({})
        logger.info("Cleared %d existing documents.", deleted.deleted_count)

        # Load SentenceTransformer model
        model = SentenceTransformer("intfloat/e5-small-v2")
        logger.info("Loaded embedding model.")

        records = []

        for cluster_id, emotion_label in EMOTION_LABELS.items():
            synonyms = EXTENDED_EMOTION_LABELS.get(emotion_label, [])[:100]
            logger.info("Processing '%s' with %d synonyms.", emotion_label, len(synonyms))

            synonym_embeddings = []
            for word in synonyms:
                embedding = model.encode(word, normalize_embeddings=True)
                synonym_embeddings.append({
                    "word": word,
                    "embedding": embedding.tolist()
                })

            record = {
                "cluster": cluster_id,
                "emotion_level": emotion_label,
                "synonyms": synonym_embeddings
            }
            records.append(record)

        result = collection.insert_many(records)
        logger.info("Inserted %d emotional level documents.", len(result.inserted_ids))

        collection.create_index("cluster", unique=True)
        collection.create_index("emotion_level", unique=True)
        logger.info("Created indexes on 'cluster' and 'emotion_level'.")

    except Exception as e:
        logger.error("Error during ingestion: %s", e)
    finally:
        client.close()
        logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    ingest_emotional_levels_grouped()
