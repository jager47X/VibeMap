import os
import logging
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import normalize
from config import MONGO_URI, COLLECTION, EMOTIONAL_LEVEL_COLLECTION, EMOTION_LABELS, EMOTION_COLOR_MAP

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_embeddings(config, normalize_flag=True):
    logger.info("Loading tweet embeddings...")
    client = MongoClient(MONGO_URI)
    db = client[config["db_name"]]
    collection = db[config["embedding_collection_name"]]
    docs = list(collection.find({"embedding": {"$exists": True}}))
    client.close()

    vectors, usernames, tweets_time, tweets = [], [], [], []
    for doc in docs:
        if doc.get("embedding"):
            vectors.append(doc["embedding"])
            usernames.append(doc.get("username", "Unknown"))
            tweets_time.append(doc.get("tweets_time", "Unknown"))
            tweets.append(doc.get("tweets", ""))
    vectors = np.array(vectors).astype("float32")
    if normalize_flag:
        vectors = normalize(vectors)
    logger.info("Loaded %d embeddings.", len(vectors))
    return vectors, usernames, tweets_time, tweets

def load_emotion_embeddings(config, normalize_flag=True):
    logger.info("Loading emotion level embeddings...")
    client = MongoClient(MONGO_URI)
    db = client[config["db_name"]]
    collection = db[EMOTIONAL_LEVEL_COLLECTION]
    docs = list(collection.find({"embedding": {"$exists": True}}))
    client.close()

    emotions = {}
    for doc in docs:
        cluster = doc.get("cluster")
        if cluster is not None and doc.get("embedding"):
            emb = np.array(doc["embedding"]).astype("float32")
            if normalize_flag:
                emb = normalize(emb.reshape(1, -1))[0]
            emotions[cluster] = {
                "embedding": emb,
                "spectacle_emotion_name": doc.get("spectacle_emotion_name", "Unknown"),
                "vibe_level": doc.get("vibe_level", "Unknown"),
                "visual_symbol": doc.get("visual_symbol", "Unknown"),
                "color": doc.get("color", "Unknown")
            }
    logger.info("Loaded %d emotion embeddings.", len(emotions))
    return emotions

def assign_emotions(tweet_vectors, emotion_embeddings):
    labels, emotion_details_list = [], []
    sorted_keys = sorted(emotion_embeddings.keys())
    emotion_matrix = np.stack([emotion_embeddings[k]["embedding"] for k in sorted_keys])
    
    for idx, tweet_vec in enumerate(tweet_vectors):
        sims = np.dot(emotion_matrix, tweet_vec)
        best_idx = np.argmax(sims)
        assigned_cluster = sorted_keys[best_idx]
        
        # Debug logging for cosine similarities and the assigned cluster.
        logger.debug("Tweet index %d: Cosine similarities: %s", idx, sims)
        logger.debug("Tweet index %d: Best similarity at index %d, assigned cluster: %s (score: %.4f)",
                     idx, best_idx, assigned_cluster, sims[best_idx])
        
        labels.append(assigned_cluster)
        emo = emotion_embeddings[assigned_cluster].copy()
        # Remove unused fields
        emo.pop("embedding", None)
        emo.pop("spectacle_emotion_name", None)
        emo.pop("visual_symbol", None)
        emotion_details_list.append(emo)
    
    return np.array(labels), emotion_details_list

def save_to_mongodb(tweet_vectors, usernames, labels, emotion_details_list, tweets_time, tweets, config):
    logger.info("Saving tweet records directly to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[config["db_name"]]
    collection_name = config.get("emotion_assigned_collection_name", "emotion_assigned")
    collection = db[collection_name]
    
    output_data = []
    for i, (tweet_vec, label, username, tweet_time, tweet, emo_details) in enumerate(
            zip(tweet_vectors, labels, usernames, tweets_time, tweets, emotion_details_list)):
        # Convert label to integer and use it for lookups
        assigned_cluster = int(label)
        # Look up the emotion label based on the assigned cluster
        emotion_label = EMOTION_LABELS.get(assigned_cluster, "Unknown")
        # Use the emotion label to look up the color from the color map
        emotion_color = EMOTION_COLOR_MAP.get(emotion_label, "#000000")
        
        tweet_record = {
            "index": i,
            "title": f"Tweet by {username} at {tweet_time}",
            "tweets": tweet,
            "username": username,
            "timestamp": tweet_time,
            "embeddings": tweet_vec.tolist(),  # Convert NumPy array to list
            "emotion_details": {
                "assigned_cluster": assigned_cluster,
                "EMOTION_LABELS": emotion_label,
                "EMOTION_COLOR": emotion_color
            }
        }
        
        logger.debug("Processing tweet index %d: username=%s, timestamp=%s, assigned_cluster=%d, "
                     "emotion_label=%s, emotion_color=%s",
                     i, username, tweet_time, assigned_cluster, emotion_label, emotion_color)
        output_data.append(tweet_record)
    
    logger.info("Inserting %d tweet records into collection '%s'", len(output_data), collection_name)
    result = collection.insert_many(output_data)
    client.close()
    logger.info("Records saved to MongoDB with ids %s", result.inserted_ids)

def main():
    logger.info("Available configurations:")
    for idx, config in enumerate(COLLECTION, 1):
        logger.info("%d: %s", idx, config.get("document_type", "Unknown"))
    
    selected = int(input("Enter configuration number: ").strip()) - 1
    config = COLLECTION[selected] if 0 <= selected < len(COLLECTION) else COLLECTION[0]
    
    tweet_vectors, usernames, tweets_time, tweets = load_embeddings(config)
    emotion_embeddings = load_emotion_embeddings(config)
    labels, emotion_details_list = assign_emotions(tweet_vectors, emotion_embeddings)
    save_to_mongodb(tweet_vectors, usernames, labels, emotion_details_list, tweets_time, tweets, config)

if __name__ == "__main__":
    main()
