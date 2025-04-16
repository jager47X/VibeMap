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

def load_emotion_synonyms(config):
    """
    Loads and normalizes the emotion-synonym embeddings into memory.
    This is usually small enough to fit in memory without an issue.
    """
    logger.info("Loading emotion-level synonyms...")
    client = MongoClient(MONGO_URI)
    db = client[config["db_name"]]
    collection = db[EMOTIONAL_LEVEL_COLLECTION]

    emotion_synonyms = {}
    emotion_metadata = {}

    # Find all docs where synonyms exist
    for doc in collection.find({"synonyms": {"$exists": True}}):
        cluster = doc.get("cluster")
        label = doc.get("emotion_level", f"Cluster {cluster}")
        # Extract and normalize synonyms for this cluster
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

    # Sort clusters to maintain consistent order
    sorted_clusters = sorted(emotion_synonyms.keys())
    logger.info("Loaded synonyms for %d emotion clusters.", len(sorted_clusters))
    return emotion_synonyms, emotion_metadata, sorted_clusters

def assign_emotions_streaming(config, normalize_flag=True, log_interval=100):
    """
    Streams through the tweet embeddings from MongoDB, assigns emotions,
    and writes the results to the assigned-tweets collection.
    """

    # 1. Load all emotion synonym embeddings into memory
    emotion_synonyms, emotion_metadata, sorted_clusters = load_emotion_synonyms(config)

    # 2. Set up reading from "embedding_collection_name"
    client = MongoClient(MONGO_URI)
    db = client[config["db_name"]]
    source_collection = db[config["embedding_collection_name"]]
    target_collection = db[EMOTION_ASSIGNED_TWEETS_COLLECTION]

    logger.info("Starting the streaming assignment of emotions...")
    cursor = source_collection.find({"embedding": {"$exists": True}})

    count = 0
    for doc in cursor:
        embedding = doc.get("embedding")
        if not embedding:
            continue

        # Convert to array
        tweet_vec = np.array(embedding, dtype=np.float32)
        # Optionally normalize
        if normalize_flag:
            # shape it to 2D before normalize(), then flatten back
            tweet_vec = tweet_vec.reshape(1, -1)
            tweet_vec = normalize(tweet_vec)[0]

        # Calculate medians for each cluster
        medians = []
        all_cluster_scores = {}
        for cluster in sorted_clusters:
            synonym_matrix = emotion_synonyms[cluster]
            sims = np.dot(synonym_matrix, tweet_vec)
            median_sim = np.median(sims)
            medians.append(median_sim)
            # For logging/debugging, track top similarities
            all_cluster_scores[cluster] = {
                "median": median_sim,
                "top_similarities": np.round(np.sort(sims)[-5:][::-1], 4).tolist()
            }

        # Determine best cluster
        best_idx = int(np.argmax(medians))
        assigned_cluster = sorted_clusters[best_idx]
        assigned_emotion = emotion_metadata[assigned_cluster]

        # Construct record for the assigned tweets
        username = doc.get("username", "Unknown")
        tweet_time = doc.get("tweets_time", "Unknown")
        tweet_text = doc.get("tweets", "")

        tweet_record = {
            "index": count,
            "title": f"Tweet by {username} at {tweet_time}",
            "tweets": tweet_text,
            "username": username,
            "timestamp": tweet_time,
            "embeddings": tweet_vec.tolist(),
            "emotion_details": {
                "assigned_cluster": assigned_cluster,
                "EMOTION_LABELS": assigned_emotion["label"],
                "EMOTION_COLOR": assigned_emotion["color"],
                "all_medians": {
                    emotion_metadata[c]["label"]: round(float(all_cluster_scores[c]["median"]), 4)
                    for c in sorted_clusters
                },
                "top_similarities": {
                    emotion_metadata[c]["label"]: [
                        round(float(x), 4) for x in all_cluster_scores[c]["top_similarities"]
                    ]
                    for c in sorted_clusters
                }
            }
        }

        # Insert into target collection
        target_collection.insert_one(tweet_record)

        count += 1
        # Log progress occasionally
        if count % log_interval == 0:
            logger.info("Processed %d tweets...", count)

    client.close()
    logger.info("Finished assigning emotions to %d tweets.", count)

def main():
    logger.info("Available configurations:")
    for idx, config in enumerate(COLLECTION, 1):
        logger.info("%d: %s", idx, config.get("document_type", "Unknown"))

    selected = int(input("Enter configuration number: ").strip()) - 1
    config = COLLECTION[selected] if 0 <= selected < len(COLLECTION) else COLLECTION[0]

    # Perform streaming assignment
    assign_emotions_streaming(config, normalize_flag=True)

if __name__ == "__main__":
    main()
