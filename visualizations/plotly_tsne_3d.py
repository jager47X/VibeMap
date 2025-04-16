import os
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from pymongo import MongoClient
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import (
    MONGO_URI,
    DB_NAME,
    EMOTION_ASSIGNED_TWEETS_COLLECTION,
    EMOTION_COLOR_MAP,
    COLLECTION
)

# === Plotly Default ===
pio.templates.default = "plotly_white"

# === Logger Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_mongo_data(collection_name, limit=None):
    logger.info("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    cursor = db[collection_name].find()
    if limit:
        cursor = cursor.limit(limit)
    data = list(cursor)

    logger.info("Loaded %d documents from '%s'", len(data), collection_name)
    client.close()
    return data


def compute_tsne(embeddings, max_itr_input,random_state=42):
    logger.info("Performing PCA...")
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    logger.info("Running t-SNE...")
    tsne = TSNE(n_components=3, perplexity=30, max_iter=max_itr_input, init='pca', random_state=random_state, verbose=2)
    return tsne.fit_transform(reduced)


def prepare_dataframe(data,max_itr_input=1000):
    embeddings = np.array([d["embeddings"] for d in data])
    tsne_coords = compute_tsne(embeddings,max_itr_input)

    records = []
    for idx, (doc, coord) in enumerate(zip(data, tsne_coords)):
        emotion = doc.get("emotion_details", {}).get("EMOTION_LABELS", "unknown")
        cluster = doc.get("emotion_details", {}).get("assigned_cluster", "N/A")
        record = {
            "index": idx,
            "title": doc.get("title", f"Document {idx}"),
            "tweets": doc.get("tweets", ""),
            "username": doc.get("username", "unknown").strip().lower(),
            "timestamp": pd.to_datetime(doc.get("timestamp", None), errors="coerce"),
            "emotion": emotion,
            "cluster": cluster,
            "x": coord[0],
            "y": coord[1],
            "z": coord[2]
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.dropna(subset=["timestamp"], inplace=True)
    df["time_bucket"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    logger.info("Prepared DataFrame with %d rows", len(df))
    return df


def build_plot(df, output_html):
    import plotly.graph_objects as go

    logger.info("Building animated 3D Plot...")

    df = df.sort_values("timestamp")
    unique_dates = sorted(df["time_bucket"].unique())

    # Create frames for animation
    all_frames = []
    for date in unique_dates:
        subset = df[df["time_bucket"] == date]
        frame = {
            "name": date,
            "data": [go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode="markers",
                marker=dict(color=subset["emotion"].map(EMOTION_COLOR_MAP)),
                text=subset["title"],
                customdata=np.stack([
                    subset["username"],
                    subset["timestamp"].astype(str),
                    subset["tweets"]
                ], axis=-1),
                hovertemplate=(
                    "<b>Title:</b> %{text}<br>"
                    "<b>Username:</b> %{customdata[0]}<br>"
                    "<b>Time:</b> %{customdata[1]}<br>"
                    "<b>Tweet:</b> %{customdata[2]}<extra></extra>"
                ),
                showlegend=False
            )]
        }
        all_frames.append(frame)

    # ALL TIME frame
    all_time_trace = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        marker=dict(color=df["emotion"].map(EMOTION_COLOR_MAP)),
        text=df["title"],
        customdata=np.stack([
            df["username"],
            df["timestamp"].astype(str),
            df["tweets"]
        ], axis=-1),
        hovertemplate=(
            "<b>Title:</b> %{text}<br>"
            "<b>Username:</b> %{customdata[0]}<br>"
            "<b>Time:</b> %{customdata[1]}<br>"
            "<b>Tweet:</b> %{customdata[2]}<extra></extra>"
        ),
        showlegend=False
    )

    all_time_frame = {"name": "ALL TIME", "data": [all_time_trace]}

    # Slider steps (ALL TIME first)
    steps = [{
        "args": [[frame["name"]], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
        "label": frame["name"],
        "method": "animate"
    } for frame in [all_time_frame] + all_frames]

    # Dropdown filters
    usernames = sorted(df["username"].unique())
    emotions = sorted(df["emotion"].unique())

    username_buttons = [{
        "label": "All Users",
        "method": "restyle",
        "args": [{"x": [df["x"]], "y": [df["y"]], "z": [df["z"]],
                  "text": [df["title"]],
                  "customdata": [np.stack([df["username"], df["timestamp"].astype(str), df["tweets"]], axis=-1)],
                  "marker.color": [df["emotion"].map(EMOTION_COLOR_MAP)]}]
    }] + [{
        "label": user,
        "method": "restyle",
        "args": [{
            "x": [df[df["username"] == user]["x"]],
            "y": [df[df["username"] == user]["y"]],
            "z": [df[df["username"] == user]["z"]],
            "text": [df[df["username"] == user]["title"]],
            "customdata": [np.stack([
                df[df["username"] == user]["username"],
                df[df["username"] == user]["timestamp"].astype(str),
                df[df["username"] == user]["tweets"]
            ], axis=-1)],
            "marker.color": [df[df["username"] == user]["emotion"].map(EMOTION_COLOR_MAP)]
        }]
    } for user in usernames]

    emotion_buttons = [{
        "label": "All Emotions",
        "method": "restyle",
        "args": [{"x": [df["x"]], "y": [df["y"]], "z": [df["z"]],
                  "text": [df["title"]],
                  "customdata": [np.stack([df["username"], df["timestamp"].astype(str), df["tweets"]], axis=-1)],
                  "marker.color": [df["emotion"].map(EMOTION_COLOR_MAP)]}]
    }] + [{
        "label": emotion,
        "method": "restyle",
        "args": [{
            "x": [df[df["emotion"] == emotion]["x"]],
            "y": [df[df["emotion"] == emotion]["y"]],
            "z": [df[df["emotion"] == emotion]["z"]],
            "text": [df[df["emotion"] == emotion]["title"]],
            "customdata": [np.stack([
                df[df["emotion"] == emotion]["username"],
                df[df["emotion"] == emotion]["timestamp"].astype(str),
                df[df["emotion"] == emotion]["tweets"]
            ], axis=-1)],
            "marker.color": [df[df["emotion"] == emotion]["emotion"].map(EMOTION_COLOR_MAP)]
        }]
    } for emotion in emotions]

    # Static color legend (emotion labels)
    emotion_legend_annotations = [{
        "x": 1.15,
        "y": 0.9 - 0.05 * i,
        "xref": "paper",
        "yref": "paper",
        "text": f'<span style="color:{EMOTION_COLOR_MAP[e]};"><b>{e}</b></span>',
        "showarrow": False,
        "font": {"size": 12}
    } for i, e in enumerate(EMOTION_COLOR_MAP.keys())]

    # Final layout
    layout = go.Layout(
        title=f"3D t-SNE Emotion Visualization of {COLLECTION[0]['document_type']}",
        updatemenus=[
            {
                "buttons": username_buttons,
                "direction": "down",
                "x": 1.02,
                "y": 1.03,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True
            },
            {
                "buttons": emotion_buttons,
                "direction": "down",
                "x": 1.20,
                "y": 1.03,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True
            }
        ],
        sliders=[{
            "active": 0,
            "x": 0.01,
            "currentvalue": {"prefix": "Date: "},
            "pad": {"t": 40},
            "steps": steps
        }],
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, showbackground=False),
            yaxis=dict(visible=False, showgrid=False, showbackground=False),
            zaxis=dict(visible=False, showgrid=False, showbackground=False),
            aspectmode="cube"
        ),
        annotations=emotion_legend_annotations,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="white",
        font=dict(color="black")
    )

    # Final figure
    fig = go.Figure(
        data=all_time_frame["data"],
        layout=layout,
        frames=[all_time_frame] + all_frames
    )

    logger.info("Saving plot to: %s", output_html)
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html, auto_open=True)


def main():
    limit_input = input("Enter max number of records to process (press Enter for no limit): ")
    try:
        limit = int(limit_input)
    except ValueError:
        limit = None

    data = load_mongo_data(EMOTION_ASSIGNED_TWEETS_COLLECTION, limit)
    if not data:
        logger.warning("No data loaded. Exiting.")
        return
    
    max_itr_input = input("Enter max number of interate to process (press Enter for no limit, min 250): ")
    max_itr_input = int(max_itr_input)
    if max_itr_input < 250:
        max_itr_input = 250
        logger.info("Max iterations set to minimum value of 250.")
        
    df = prepare_dataframe(data,max_itr_input)
    output_html = os.path.join("Data", "visualizations_outputs", f"{EMOTION_ASSIGNED_TWEETS_COLLECTION}_tsne_plot.html")
    build_plot(df, output_html)


if __name__ == "__main__":
    main()