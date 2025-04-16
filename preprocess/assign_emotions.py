import logging
import numpy as np
import pymongo
from pymongo import MongoClient
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import (
    MONGO_URI,
    DB_NAME,
    COLLECTION_NAME,
    LABEL_COLLECTION,
    EMOTIONAL_LEVEL_COLLECTION,
    EMOTION_LABELS,
    EMOTION_COLOR_MAP,
    EMOTION_ASSIGNED_TWEETS_COLLECTION
)

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # -- Console handler: leave it as-is, prints everything --
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # -- File handler: overwrite each run, but only evaluation lines --
    fh = logging.FileHandler("evaluation.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Define a filter that only lets through evaluation messages
    class EvalFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # adjust these conditions to match exactly your evaluation logs
            return (
                msg.startswith("Stage") or
                msg.startswith("Test soft‑accuracy") or
                msg.startswith("Final soft‑accuracy") or
                msg.startswith("Report:")
            )

    fh.addFilter(EvalFilter())
    logger.addHandler(fh)

    return logger

def score_by_error(true, pred):
    """Soft-accuracy: exact=1.0, else 0.8/0.6/0.4/0.2/0.0 by |Δcluster|."""
    err = abs(int(pred) - int(true))
    return {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}.get(err, 0.0)

def score_vector(y_true, y_pred):
    return np.array([score_by_error(t, p) for t, p in zip(y_true, y_pred)])
logger = setup_logger()
def load_emotion_prototypes(db):
    """Load one normalized prototype vector per cluster."""
    logger.info("Loading prototypes from %s.%s", DB_NAME, EMOTIONAL_LEVEL_COLLECTION)
    coll = db[EMOTIONAL_LEVEL_COLLECTION]
    protos = {}

    for doc in coll.find():
        c = doc.get("cluster")
        if "embedding" in doc:
            v = np.array(doc["embedding"], dtype=np.float32)
        else:
            embs = [
                np.array(s["embedding"], dtype=np.float32)
                for s in doc.get("synonyms", [])
                if "embedding" in s
            ]
            if not embs:
                continue
            v = normalize(np.stack(embs), axis=1).mean(axis=0)

        v = v / np.linalg.norm(v)
        protos[c] = v

    clusters = sorted(protos)
    logger.info("  -> %d prototypes loaded", len(clusters))
    return protos, clusters

def load_dataset(db):
    """Load X, y (and docs) by joining embeddings with manual labels, fallback to true_cluster."""
    logger.info("Loading data: %s.%s <-> %s.%s",
                DB_NAME, LABEL_COLLECTION, DB_NAME, COLLECTION_NAME)
    X, y, docs = [], [], []

    labels = list(db[LABEL_COLLECTION].find({"label_idx": {"$exists": True}}))
    logger.info("  -> Found %d labels", len(labels))
    for lbl in labels:
        emb = db[COLLECTION_NAME].find_one({"_id": lbl["_id"]})
        if not emb or "embedding" not in emb:
            continue
        vec = normalize(
            np.array(emb["embedding"], dtype=np.float32).reshape(1, -1)
        )[0]
        X.append(vec); y.append(int(lbl["label_idx"])); docs.append(emb)

    if not X:
        logger.warning("  -> No joined docs; falling back on true_cluster")
        for doc in db[COLLECTION_NAME].find({
                "embedding": {"$exists": True},
                "true_cluster": {"$exists": True}
            }):
            vec = normalize(
                np.array(doc["embedding"], dtype=np.float32).reshape(1, -1)
            )[0]
            X.append(vec); y.append(int(doc["true_cluster"])); docs.append(doc)

    if not X:
        logger.error("No data found! Exiting.")
        raise SystemExit

    X = np.stack(X)
    y = np.array(y)
    logger.info("  -> Loaded %d samples", len(y))
    return X, y, docs

def proto_predict(X, protos, clusters):
    M = np.stack([protos[c] for c in clusters])
    sims = X.dot(M.T)
    return np.array([clusters[i] for i in sims.argmax(axis=1)])

def assign_all_and_save(db, clf, protos, clusters):
    """Apply model to every embedding-doc and upsert emotion details."""
    logger.info("Labeling all docs in %s.%s", DB_NAME, COLLECTION_NAME)
    emb_coll = db[COLLECTION_NAME]
    out_coll = db[EMOTION_ASSIGNED_TWEETS_COLLECTION]
    bulk, count = [], 0

    for doc in emb_coll.find({"embedding": {"$exists": True}}):
        vec = normalize(
            np.array(doc["embedding"], dtype=np.float32).reshape(1, -1)
        )[0]

        proto_cl = proto_predict(vec[None, :], protos, clusters)[0]
        sup_cl   = int(clf.predict(vec[None, :])[0])

        enriched = doc.copy()
        enriched["emotion_details"] = {
            "prototype_cluster": int(proto_cl),
            "assigned_cluster":   sup_cl,
            "label":              EMOTION_LABELS[sup_cl],
            "color":              EMOTION_COLOR_MAP[EMOTION_LABELS[sup_cl]]
        }

        bulk.append(pymongo.ReplaceOne({"_id": doc["_id"]}, enriched, upsert=True))
        count += 1

        if len(bulk) >= 500:
            try:
                res = out_coll.bulk_write(bulk)
                logger.info("  -> Upserted %d docs", res.upserted_count + res.modified_count)
            except Exception as e:
                logger.error("Bulk write failed: %s", e)
            bulk.clear()

    if bulk:
        try:
            res = out_coll.bulk_write(bulk)
            logger.info("  -> Upserted %d docs", res.upserted_count + res.modified_count)
        except Exception as e:
            logger.error("Final bulk write failed: %s", e)

    logger.info("Finished labeling %d docs.", count)

def main():
    # ---- Single client for whole run
    with MongoClient(MONGO_URI) as client:
        db = client[DB_NAME]

        protos, clusters = load_emotion_prototypes(db)
        X, y, docs      = load_dataset(db)

        # train/test split (now includes docs)
        X_tr, X_te, y_tr, y_te, docs_tr, docs_te = train_test_split(
            X, y, docs,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # Stage 1: Prototype Matching
        p_te = proto_predict(X_te, protos, clusters)
        logger.info("Stage 1: Prototype Matching soft-accuracy: %.3f", score_vector(y_te, p_te).mean())
        logger.info("Report:\n%s",
                    classification_report(y_te, p_te,
                                          labels=clusters,
                                          target_names=[EMOTION_LABELS[c] for c in clusters],
                                          zero_division=0))

        # Stage 2: Supervised Learning
        err1_tr    = score_vector(y_tr, proto_predict(X_tr, protos, clusters))
        weights_tr = np.clip(1.0 - err1_tr + 0.1, 0.1, None)
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                 max_iter=500, random_state=42)
        clf.fit(X_tr, y_tr, sample_weight=weights_tr)

        s_te = clf.predict(X_te)
        logger.info("Stage 2: Supervised soft-accuracy: %.3f", score_vector(y_te, s_te).mean())
        logger.info("Report:\n%s",
                    classification_report(y_te, s_te,
                                          labels=clusters,
                                          target_names=[EMOTION_LABELS[c] for c in clusters],
                                          zero_division=0))

        # Stage 3: Unsupervised Residual Correction
        err2_te    = score_vector(y_te, s_te)
        mask       = err2_te < 1.0
        X_err, y_err = X_te[mask], y_te[mask]
        final = s_te.copy()
        if len(X_err) >= len(clusters):
            km       = KMeans(n_clusters=len(clusters), random_state=42).fit(X_err)
            centers  = normalize(km.cluster_centers_, axis=1)
            proto_mat = np.stack([protos[c] for c in clusters])
            mapping  = {
                ci: clusters[centers[ci].dot(proto_mat.T).argmax()]
                for ci in range(len(clusters))
            }
            u_pred   = np.array([mapping[l] for l in km.labels_])
            for i, idx in enumerate(np.where(mask)[0]):
                if score_by_error(y_err[i], u_pred[i]) > err2_te[idx]:
                    final[idx] = u_pred[i]

            logger.info("Stage 3: Residual KMeans soft-accuracy: %.3f", score_vector(y_te, final).mean())
            logger.info("Final Report:\n%s",
                        classification_report(y_te, final,
                                              labels=clusters,
                                              target_names=[EMOTION_LABELS[c] for c in clusters],
                                              zero_division=0))
        else:
            logger.info("Stage 3: Skipped (only %d residuals, need >= %d)", len(X_err), len(clusters))

        # Apply to all tweets
        assign_all_and_save(db, clf, protos, clusters)

if __name__ == "__main__":
    main()

