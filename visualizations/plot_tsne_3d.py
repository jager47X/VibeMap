import os
import sys
import math
import logging

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from pymongo import MongoClient
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Dash ------------------------------------------------------------
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------
# Make project root importable so we can pull in config.py that sits one level up
# -----------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    MONGO_URI,
    DB_NAME,
    EMOTION_ASSIGNED_TWEETS_COLLECTION,
    EMOTION_COLOR_MAP,
)

# -----------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------
pio.templates.default = "plotly_white"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------

def safe_int(value, default):
    """Safely cast *value* to int, otherwise return *default*."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------
# Mongo helpers & dimensionality-reduction pipeline
# -----------------------------------------------------------------

def load_mongo_data(collection_name: str, limit: int | None = None):
    logger.info("Connecting to MongoDB …")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    cursor = db[collection_name].find()
    if limit:
        cursor = cursor.limit(limit)

    data = list(cursor)
    logger.info("Loaded %d documents from '%s'", len(data), collection_name)
    client.close()
    return data


def compute_tsne(embeddings: np.ndarray, n_iter: int, random_state: int = 42):
    logger.info("Running optimized t‑SNE (%d iterations) without PCA preprocessing …", n_iter)
    
    tsne = TSNE(
        n_components=3,
        perplexity=40,            # adjust based on data size
        n_iter=n_iter,
        init="pca",               # internal PCA for better init, even without explicit preprocessing
        learning_rate="auto",
        metric="cosine",          # more meaningful for embeddings
        n_jobs=-1,                # if sklearn version supports it
        random_state=random_state,
        verbose=2,
    )
    return tsne.fit_transform(embeddings)




def prepare_dataframe(raw_docs: list[dict], n_iter: int):
    """Return a tidy DataFrame ready for Plotly."""
    embeddings = np.array([d["embeddings"] for d in raw_docs])
    coords = compute_tsne(embeddings, n_iter)

    records = []
    for idx, (doc, coord) in enumerate(zip(raw_docs, coords)):
        emotion = doc.get("emotion_details", {}).get("EMOTION_LABELS", "unknown")
        cluster = doc.get("emotion_details", {}).get("assigned_cluster", "N/A")
        records.append(
            {
                "index": idx,
                "title": doc.get("title", f"Document {idx}"),
                "tweets": doc.get("tweets", ""),
                "username": doc.get("username", "unknown").strip().lower(),
                "timestamp": pd.to_datetime(doc.get("timestamp"), errors="coerce"),
                "emotion": emotion,
                "cluster": cluster,
                "x": coord[0],
                "y": coord[1],
                "z": coord[2],
            }
        )

    df = pd.DataFrame(records)
    df.dropna(subset=["timestamp"], inplace=True)
    df["time_bucket"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    logger.info("Prepared DataFrame with %d rows", len(df))
    return df


# -----------------------------------------------------------------
# Build Plotly figure
# -----------------------------------------------------------------

def build_plot(df: pd.DataFrame) -> go.Figure:
    logger.info("Building Plotly 3-D scatter …")

    df_sorted = df.sort_values("timestamp")
    unique_dates = sorted(df_sorted["time_bucket"].unique())

    # --- Frames (per-day) ----------------------------------------------------
    frames = []
    for date in unique_dates:
        subset = df_sorted[df_sorted["time_bucket"] == date]
        frames.append(
            {
                "name": date,
                "data": [
                    go.Scatter3d(
                        x=subset["x"],
                        y=subset["y"],
                        z=subset["z"],
                        mode="markers",
                        marker=dict(color=subset["emotion"].map(EMOTION_COLOR_MAP)),
                        text=subset["title"],
                        customdata=np.stack(
                            [
                                subset["username"],
                                subset["timestamp"].astype(str),
                                subset["tweets"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>Title:</b> %{text}<br>"
                            "<b>User:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>Tweet:</b> %{customdata[2]}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                ],
            }
        )

    # All-time trace & frame ---------------------------------------------------
    all_trace = go.Scatter3d(
        x=df_sorted["x"],
        y=df_sorted["y"],
        z=df_sorted["z"],
        mode="markers",
        marker=dict(color=df_sorted["emotion"].map(EMOTION_COLOR_MAP)),
        text=df_sorted["title"],
        customdata=np.stack(
            [
                df_sorted["username"],
                df_sorted["timestamp"].astype(str),
                df_sorted["tweets"],
            ],
            axis=-1,
        ),
        hovertemplate=(
            "<b>Title:</b> %{text}<br>"
            "<b>User:</b> %{customdata[0]}<br>"
            "<b>Time:</b> %{customdata[1]}<br>"
            "<b>Tweet:</b> %{customdata[2]}<extra></extra>"
        ),
        showlegend=False,
    )

    all_frame = {"name": "ALL TIME", "data": [all_trace]}

    # Slider steps -------------------------------------------------------------
    slider_steps = [
        {
            "args": [[all_frame["name"]], {"frame": {"duration": 400, "redraw": True}}],
            "label": "ALL TIME",
            "method": "animate",
        }
    ] + [
        {
            "args": [[f["name"]], {"frame": {"duration": 400, "redraw": True}}],
            "label": f["name"],
            "method": "animate",
        }
        for f in frames
    ]

    # Username & emotion dropdowns --------------------------------------------
    usernames = sorted(df_sorted["username"].unique())
    emotions = sorted(df_sorted["emotion"].unique())

    def make_restyle_button(label, mask):
        filt = df_sorted[mask]
        return {
            "label": label,
            "method": "restyle",
            "args": [
                {
                    "x": [filt["x"]],
                    "y": [filt["y"]],
                    "z": [filt["z"]],
                    "text": [filt["title"]],
                    "customdata": [
                        np.stack(
                            [
                                filt["username"],
                                filt["timestamp"].astype(str),
                                filt["tweets"],
                            ],
                            axis=-1,
                        )
                    ],
                    "marker.color": [filt["emotion"].map(EMOTION_COLOR_MAP)],
                }
            ],
        }

    username_buttons = [
        make_restyle_button("All Users", slice(None))
    ] + [
        make_restyle_button(u, df_sorted["username"] == u) for u in usernames
    ]

    emotion_buttons = [
        make_restyle_button("All Emotions", slice(None))
    ] + [
        make_restyle_button(e, df_sorted["emotion"] == e) for e in emotions
    ]

    # Emotion legend (text with color) ----------------------------------------
    legend_ann = [
        {
            "x": 1.15,
            "y": 0.9 - 0.05 * i,
            "xref": "paper",
            "yref": "paper",
            "text": f'<span style="color:{EMOTION_COLOR_MAP[e]}"><b>{e}</b></span>',
            "showarrow": False,
        }
        for i, e in enumerate(EMOTION_COLOR_MAP.keys())
    ]

    layout = go.Layout(
        title="3-D t-SNE Emotion Map",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
        ),
        sliders=[
            {
                "active": 0,
                "x": 0.01,
                "currentvalue": {"prefix": "Date: "},
                "pad": {"t": 40},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "buttons": username_buttons,
                "direction": "down",
                "x": 1.02,
                "y": 1.03,
                "showactive": True,
            },
            {
                "buttons": emotion_buttons,
                "direction": "down",
                "x": 1.20,
                "y": 1.03,
                "showactive": True,
            },
        ],
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
        annotations=legend_ann,
    )

    fig = go.Figure(data=all_frame["data"], layout=layout, frames=[all_frame] + frames)
    return fig


# -----------------------------------------------------------------
# Dash app & layout
# -----------------------------------------------------------------
external_stylesheets = [dbc.themes.FLATLY]
app = dash.Dash(
    __name__,
    assets_folder="../Data/assets",  # relative to this file
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
app.title = "Vibe Map"

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Interval(id="countdown-interval", interval=1_000, n_intervals=0, disabled=True),
        dcc.Store(id="remaining_time_store"),
        dcc.Store(id="plot_store"),

        # Headline ------------------------------------------------------------
        dbc.Row(
            dbc.Col(html.H1("Vibe Map", className="text-center my-4"), width=12)
        ),

        # Controls ------------------------------------------------------------
        html.Div(
            id="controls-container",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Max Records (default 100):"),
                                dbc.Input(id="limit_input", type="number", placeholder="Records", min=1),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Max t-SNE Iterations (min 250):"),
                                dbc.Input(id="max_itr_input", type="number", placeholder="Iterations", min=250, value=250),
                            ],
                            width=6,
                        ),
                    ],
                    class_name="mb-3",
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Generate Plot",
                            id="generate_button",
                            color="primary",
                            class_name="mb-3",
                            n_clicks=0,
                        ),
                        width="auto",
                        class_name="d-flex justify-content-center",
                    )
                ),

                # Estimated-time + spinner -----------------------------------
                dcc.Loading(
                    id="loading-spinner",
                    type="default",
                    children=html.Div(
                        [
                            html.Div(id="estimated-time", className="text-center"),
                            html.Div(id="loading-message", className="text-center mt-2"),
                        ]
                    ),
                ),

                # Logo filler -------------------------------------------------
                html.Div(
                    html.Img(
                        src="/assets/logo.png",
                        style={
                            "width": "100%",
                            "height": "80vh",
                            "objectFit": "contain",
                            "transform": "scale(0.9)",
                            "margin": "auto",
                        },
                    ),
                    style={"marginTop": "1rem"},
                ),
            ],
        ),

        # Page content (plot) --------------------------------------------------
        html.Div(id="page-content"),
    ],
    fluid=True,
)


# -----------------------------------------------------------------
# Callback: Estimated time & countdown
# -----------------------------------------------------------------
@app.callback(
    Output("estimated-time", "children"),
    Output("remaining_time_store", "data"),
    Output("countdown-interval", "disabled"),
    Output("countdown-interval", "n_intervals"),
    Output("loading-message", "children"),
    Input("limit_input", "value"),
    Input("max_itr_input", "value"),
    Input("generate_button", "n_clicks"),
    Input("countdown-interval", "n_intervals"),
    State("remaining_time_store", "data"),
    prevent_initial_call=True,
)
def update_estimated_time(limit_value, max_itr_value, n_clicks, n_intervals, remaining):
    limit = safe_int(limit_value, 100)
    max_itr = max(250, safe_int(max_itr_value, 250))
    total_seconds = max(1, math.ceil((limit / 10_000) * (max_itr / 5)))

    spinner = html.Div(
        dbc.Spinner(size="sm", color="primary", type="border"),
        className="text-center mt-2",
    )

    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    # Generate pressed -------------------------------------------------------
    if trigger == "generate_button":
        return (
            f"Estimated processing time: {total_seconds} seconds",
            total_seconds,
            False,  # enable interval
            0,      # reset interval counter
            spinner,
        )

    # Interval tick ----------------------------------------------------------
    if trigger == "countdown-interval":
        if remaining is None:
            return no_update, no_update, True, 0, no_update

        seconds_left = max(int(remaining) - 1, 0)

        if seconds_left == 0:
            done_msg = html.Div(
                "Estimation may vary by machine. Finalizing the plot…",
                className="text-center mt-2",
            )
            return (
                "Estimated processing time: 0 seconds",
                0,
                True,
                0,
                done_msg,
            )

        return (
            f"Estimated processing time: {seconds_left} seconds",
            seconds_left,
            False,
            no_update,
            spinner,
        )

    # Limit / iteration changed while idle -----------------------------------
    return (
        f"Estimated processing time: {total_seconds} seconds",
        None,
        True,
        0,
        "",
    )


# -----------------------------------------------------------------
# Callback: Generate plot & redirect
# -----------------------------------------------------------------
@app.callback(
    Output("plot_store", "data"),
    Output("url", "pathname"),
    Input("generate_button", "n_clicks"),
    State("limit_input", "value"),
    State("max_itr_input", "value"),
)

def generate_and_redirect(n_clicks, limit_value, max_itr_value):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    limit = safe_int(limit_value, 100)
    max_itr = max(250, safe_int(max_itr_value, 250))

    docs = load_mongo_data(EMOTION_ASSIGNED_TWEETS_COLLECTION, limit)
    if not docs:
        logger.warning("No documents found – staying on landing page.")
        return None, "/"

    df = prepare_dataframe(docs, max_itr)
    fig = build_plot(df)

    return fig.to_plotly_json(), "/plot"


# -----------------------------------------------------------------
# Callbacks: Show / hide controls & render page content
# -----------------------------------------------------------------
@app.callback(Output("controls-container", "style"), Input("url", "pathname"))
def toggle_controls(pathname):
    return {"display": "none"} if pathname == "/plot" else {"display": "block"}


@app.callback(Output("page-content", "children"), Input("url", "pathname"), State("plot_store", "data"))
def render_page(pathname, plot_json):
    if pathname == "/plot" and plot_json is not None:
        fig = go.Figure(plot_json)
        return html.Div(
            [
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Regenerate the Plot",
                            id="regenerate_button",
                            color="primary",
                            n_clicks=0,
                        ),
                        width={"size": 4, "offset": 4},
                        className="text-center",
                    )
                ),
                dcc.Graph(
                    id="tsne-3d-graph",
                    figure=fig,
                    config={"displayModeBar": False},
                    style={"height": "100vh", "width": "100%"},
                ),
                html.Div(
                    "Plot generation completed",
                    className="text-center mt-3",
                ),
            ]
        )
    return ""


# -----------------------------------------------------------------
# Callback: regenerate button → redirect to landing
# -----------------------------------------------------------------
@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("regenerate_button", "n_clicks"),
    prevent_initial_call=True,
)
def regenerate(n):
    return "/" if n else no_update


# -----------------------------------------------------------------
# Main entry point – open browser & run server
# -----------------------------------------------------------------
if __name__ == "__main__":
    import webbrowser

    port = 8050
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)
