import os
import logging
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import normalize
from config import (
    MONGO_URI, COLLECTION,
    EMOTIONAL_LEVEL_COLLECTION, EMOTION_LABELS,
    EMOTION_COLOR_MAP, EMOTION_ASSIGNED_TWEETS_COLLECTION
)

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 1000
OUTPUT_DIR = "Data/tweet_batches_npz"

def load_emotion_synonyms(config):
    logger.info("Loading emotion-level synonyms...")
    client = MongoClient(MONGO_URI)
    db = client[config["db_name"]]
    collection = db[EMOTIONAL_LEVEL_COLLECTION]

    emotion_synonyms = {}
    emotion_metadata = {}

    for doc in collection.find({"synonyms": {"$exists": True}}):
        cluster = doc.get("cluster")
        label = doc.get("emotion_level", f"Cluster {cluster}")
        synonyms = [np.array(s["embedding"], dtype=np.float32)
                    for s in doc["synonyms"] if "embedding" in s]
        if synonyms:
            emb_matrix = normalize(np.stack(synonyms, axis=0))
            emotion_synonyms[cluster] = emb_matrix
            emotion_metadata[cluster] = {
                "color": doc.get("color", "#000000"),
                "label": label
            }

    client.close()
    sorted_clusters = sorted(emotion_synonyms.keys())
    logger.info("Loaded synonyms for %d emotion clusters.", len(sorted_clusters))
    return emotion_synonyms, emotion_metadata, sorted_clusters


def save_npz_batch(batch_vectors, batch_metadata, batch_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"batch_{batch_id}.npz"),
        embeddings=np.array(batch_vectors, dtype=np.float32),
        metadata=np.array(batch_metadata, dtype=object)
    )
    logger.info(f"Saved batch {batch_id} with {len(batch_vectors)} tweets.")


def assign_emotions_streaming_batched(config, normalize_flag=True, log_interval=100):
    emotion_synonyms, emotion_metadata, sorted_clusters = load_emotion_synonyms(config)

    client = MongoClient(MONGO_URI)
    db = client[config["db_name"]]
    source_collection = db[config["embedding_collection_name"]]

    logger.info("Starting memory-efficient emotion assignment...")

    cursor = source_collection.find({"embedding": {"$exists": True}})
    count = 0
    batch_id = 0
    batch_vectors = []
    batch_metadata = []

    for doc in cursor:
        embedding = doc.get("embedding")
        if not embedding:
            continue

        tweet_vec = np.array(embedding, dtype=np.float32)
        if normalize_flag:
            tweet_vec = normalize(tweet_vec.reshape(1, -1))[0]

        medians = []
        all_cluster_scores = {}
        for cluster in sorted_clusters:
            synonym_matrix = emotion_synonyms[cluster]
            sims = np.dot(synonym_matrix, tweet_vec)
            median_sim = np.median(sims)
            medians.append(median_sim)
            all_cluster_scores[cluster] = {
                "median": median_sim,
                "top_similarities": np.round(np.sort(sims)[-5:][::-1], 4).tolist()
            }

        best_idx = int(np.argmax(medians))
        assigned_cluster = sorted_clusters[best_idx]
        assigned_emotion = emotion_metadata[assigned_cluster]

        tweet_data = {
            "index": count,
            "username": doc.get("username", "Unknown"),
            "timestamp": doc.get("tweets_time", "Unknown"),
            "tweets": doc.get("tweets", ""),
            "embedding": tweet_vec,
            "emotion": {
                "label": assigned_emotion["label"],
                "color": assigned_emotion["color"],
                "cluster": assigned_cluster,
                "medians": {
                    emotion_metadata[c]["label"]: round(float(all_cluster_scores[c]["median"]), 4)
                    for c in sorted_clusters
                },
                "top_similarities": {
                    emotion_metadata[c]["label"]: all_cluster_scores[c]["top_similarities"]
                    for c in sorted_clusters
                }
            }
        }

        batch_vectors.append(tweet_vec)
        batch_metadata.append(tweet_data)

        count += 1
        if count % log_interval == 0:
            logger.info("Processed %d tweets...", count)

        if len(batch_vectors) >= BATCH_SIZE:
            save_npz_batch(batch_vectors, batch_metadata, batch_id)
            batch_vectors, batch_metadata = [], []
            batch_id += 1

    if batch_vectors:
        save_npz_batch(batch_vectors, batch_metadata, batch_id)

    client.close()
    logger.info("Finished processing %d tweets.", count)


def main():
    logger.info("Available configurations:")
    for idx, config in enumerate(COLLECTION, 1):
        logger.info("%d: %s", idx, config.get("document_type", "Unknown"))

    selected = int(input("Enter configuration number: ").strip()) - 1
    config = COLLECTION[selected] if 0 <= selected < len(COLLECTION) else COLLECTION[0]

    assign_emotions_streaming_batched(config, normalize_flag=True)


if __name__ == "__main__":
    main()
