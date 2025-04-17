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

# Add the project root to sys.path to import config from there.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants from the project root.
from config import (
    MONGO_URI,
    DB_NAME,
    EMOTION_ASSIGNED_TWEETS_COLLECTION,
    EMOTION_COLOR_MAP,
    COLLECTION,
)

# Set default Plotly template
pio.templates.default = "plotly_white"

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Instantiate the Dash App ---
import dash
from dash import (
    dcc,
    html,
    Input,
    Output,
    State,
    callback_context,
    no_update,
)
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.FLATLY]
app = dash.Dash(
    __name__,
    assets_folder="../Data/assets",  # Updated relative path from visualizations/ folder to Data/assets folder.
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
app.title = "Vibe Map"

# ================
# Data helpers
# ================

def load_mongo_data(collection_name, limit=None):
    logger.info("Connecting to MongoDB…")
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
    """PCA‑>t‑SNE helper so we can visualise quickly"""
    logger.info("Performing PCA…")
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    logger.info("Running t‑SNE…")
    tsne = TSNE(
        n_components=3,
        perplexity=30,
        n_iter=n_iter,
        init="pca",
        random_state=random_state,
        verbose=2,
    )
    return tsne.fit_transform(reduced)


def prepare_dataframe(data, max_itr_input=1000):
    embeddings = np.asarray([d["embeddings"] for d in data])
    tsne_coords = compute_tsne(embeddings, max_itr_input)

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
            "z": coord[2],
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)
    df.dropna(subset=["timestamp"], inplace=True)
    df["time_bucket"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    logger.info("Prepared DataFrame with %d rows", len(df))
    return df


# -------------
# Build 3‑D fig
# -------------

def build_plot(df: pd.DataFrame):
    logger.info("Building animated 3‑D plot…")
    df = df.sort_values("timestamp")
    unique_dates = sorted(df["time_bucket"].unique())

    frames = []
    for date in unique_dates:
        subset = df[df["time_bucket"] == date]
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
                            "<b>Username:</b> %{customdata[0]}<br>"
                            "<b>Time:</b> %{customdata[1]}<br>"
                            "<b>Tweet:</b> %{customdata[2]}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                ],
            }
        )

    # “All‑time” baseline frame
    base_trace = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        marker=dict(color=df["emotion"].map(EMOTION_COLOR_MAP)),
        text=df["title"],
        customdata=np.stack(
            [df["username"], df["timestamp"].astype(str), df["tweets"]], axis=-1
        ),
        hovertemplate=(
            "<b>Title:</b> %{text}<br>"
            "<b>Username:</b> %{customdata[0]}<br>"
            "<b>Time:</b> %{customdata[1]}<br>"
            "<b>Tweet:</b> %{customdata[2]}<extra></extra>"
        ),
        showlegend=False,
    )
    all_time_frame = {"name": "ALL TIME", "data": [base_trace]}

    steps = [
        {
            "args": [[frame["name"]], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
            "label": frame["name"],
            "method": "animate",
        }
        for frame in [all_time_frame] + frames
    ]

    usernames = sorted(df["username"].unique())
    emotions = sorted(df["emotion"].unique())

    def _btn(label, _df):
        return {
            "label": label,
            "method": "restyle",
            "args": [
                {
                    "x": [_df["x"]],
                    "y": [_df["y"]],
                    "z": [_df["z"]],
                    "text": [_df["title"]],
                    "customdata": [
                        np.stack([
                            _df["username"],
                            _df["timestamp"].astype(str),
                            _df["tweets"],
                        ], axis=-1)
                    ],
                    "marker.color": [_df["emotion"].map(EMOTION_COLOR_MAP)],
                }
            ],
        }

    username_buttons = [_btn("All Users", df)] + [_btn(user, df[df["username"] == user]) for user in usernames]
    emotion_buttons = [_btn("All Emotions", df)] + [_btn(e, df[df["emotion"] == e]) for e in emotions]

    emotion_legend_annotations = [
        {
            "x": 1.15,
            "y": 0.9 - 0.05 * i,
            "xref": "paper",
            "yref": "paper",
            "text": f'<span style="color:{EMOTION_COLOR_MAP[e]};"><b>{e}</b></span>',
            "showarrow": False,
            "font": {"size": 12},
        }
        for i, e in enumerate(EMOTION_COLOR_MAP)
    ]

    layout = go.Layout(
        title=f"3‑D t‑SNE Emotion Visualisation of {COLLECTION[0]['document_type']}",
        updatemenus=[
            {
                "buttons": username_buttons,
                "direction": "down",
                "x": 1.02,
                "y": 1.03,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True,
            },
            {
                "buttons": emotion_buttons,
                "direction": "down",
                "x": 1.20,
                "y": 1.03,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True,
            },
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.01,
                "currentvalue": {"prefix": "Date: "},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
        ),
        annotations=emotion_legend_annotations,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="white",
        font=dict(color="black"),
    )

    return go.Figure(data=all_time_frame["data"], layout=layout, frames=[all_time_frame] + frames)


# =============================
#     LAYOUT
# =============================
app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="remaining_time_store"),
        dcc.Store(id="plot_store"),
        dcc.Interval(id="countdown-interval", interval=1000, disabled=True),

        # Controls
        html.Div(
            id="controls-container",
            children=[
                dbc.Row(dbc.Col(html.H1("Vibe Map", className="text-center my-4"))),
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Label("Max Records (default 100):"),
                            dbc.Input(id="limit_input", type="number", min=1),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Max t-SNE Iterations (min 250):"),
                            dbc.Input(id="max_itr_input", type="number", min=250, value=250),
                        ], width=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Button("Generate Plot", id="generate_button", color="primary", n_clicks=0),
                        className="d-flex justify-content-center",
                    )
                ),
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
                # Decorative image
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
        html.Div(id="page-content"),
    ],
    fluid=True,
)

# =========================================================
#   Callback: Estimated time + countdown spinner            
# =========================================================
app.callback(
    Output("estimated-time", "children"),
    Output("remaining_time_store", "data"),
    Output("countdown-interval", "disabled"),
    Output("countdown-interval", "n_intervals"),
    Output("loading-message", "children"),
    Input("limit_input", "value"),
    Input("max_itr_input", "value"),
    Input("generate_button", "n_clicks"),
    Input("countdown-interval", "n_intervals"),
    Input("url", "pathname"),
    State("remaining_time_store", "data"),
)(
    lambda limit_value, max_itr_value, n_clicks, n_intervals, pathname, remaining: _update_timer(
        limit_value,
        max_itr_value,
        n_clicks,
        n_intervals,
        pathname,
        remaining,
    )
)


def _update_timer(limit_value, max_itr_value, n_clicks, n_intervals, pathname, remaining):
    """Keeps the timer / spinner in sync with user input and backend status."""
    from dash import html
    import dash_bootstrap_components as dbc

    ctx_prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    # 1️⃣ Normalise user input ---------------------------------------------
    try:
        limit = int(limit_value) if limit_value else 100
    except (TypeError, ValueError):
        limit = 100

    try:
        max_itr = int(max_itr_value) if max_itr_value else 250
        if max_itr < 250:
            max_itr = 250
    except (TypeError, ValueError):
        max_itr = 250

    est_seconds = max(1, math.ceil((limit / 10000) * (max_itr / 5)))

    def _fmt(sec: int):
        if sec >= 3600:
            return f"{sec // 3600}h {sec % 3600 // 60}m {sec % 60}s"
        if sec >= 60:
            return f"{sec // 60}m {sec % 60}s"
        return f"{sec}s"

    spinner = dbc.Spinner(size="sm", color="primary", type="border")

    # 2️⃣ Auto‑cancel once we are on /plot (backend finished) ---------------
    if pathname == "/plot":
        return (
            "Estimated processing time: 0s",
            0,
            True,
            0,
            html.Div("Plot finished ✅", className="text-center text-success"),
        )

    # Are we currently counting down?
    active = isinstance(remaining, (int, float)) and remaining > 0

    # 3️⃣ Handle Generate click -------------------------------------------
    if "generate_button" in ctx_prop:
        return (
            f"Estimated processing time: {_fmt(est_seconds)}",
            est_seconds,
            False,  # enable interval
            0,       # reset n_intervals so the timer starts from scratch
            spinner,
        )

    # 4️⃣ Handle countdown tick -------------------------------------------
    if "countdown-interval" in ctx_prop and active:
        try:
            remaining_int = int(remaining)
        except (TypeError, ValueError):
            remaining_int = est_seconds
        remaining_int = max(remaining_int - 1, 0)

        if remaining_int == 0:
            return (
                "Estimated processing time: 0s",
                0,
                True,  # stop interval
                n_intervals,  # keep the tick count as‑is
                html.Div(
                    "Finalising the plot…", className="text-center text-secondary"
                ),
            )
        else:
            return (
                f"Estimated processing time: {_fmt(remaining_int)}",
                remaining_int,
                False,
                n_intervals,
                spinner,
            )

    # 5️⃣ Re‑compute estimate when inputs change & no run active -----------
    if ("limit_input" in ctx_prop or "max_itr_input" in ctx_prop) and not active:
        return (
            f"Estimated processing time: {_fmt(est_seconds)}",
            None,
            True,
            0,
            "",
        )

    # 6️⃣ Default (first page load) ---------------------------------------
    if not active:
        return (
            f"Estimated processing time: {_fmt(est_seconds)}",
            None,
            True,
            0,
            "",
        )

    return no_update, no_update, no_update, no_update, no_update


# =========================================================
#   Callback: generate → plot_store + redirect             
# =========================================================
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

    # Sanitize inputs
    try:
        limit = int(limit_value) if limit_value else 100
    except (TypeError, ValueError):
        limit = 100

    try:
        max_itr = int(max_itr_value) if max_itr_value else 250
        if max_itr < 250:
            max_itr = 250
    except (TypeError, ValueError):
        max_itr = 250

    data = load_mongo_data(EMOTION_ASSIGNED_TWEETS_COLLECTION, limit)
    if not data:
        logger.warning("No data loaded – staying on home page")
        return no_update, "/"

    df = prepare_dataframe(data, max_itr)
    fig = build_plot(df)

    # Immediately save figure → skips extra waiting on frontend
    return fig.to_plotly_json(), "/plot"


# =========================================================
#   Hide controls once the graph page loads                 
# =========================================================
@app.callback(Output("controls-container", "style"), Input("url", "pathname"))
def toggle_controls(path):
    return {"display": "none"} if path == "/plot" else {"display": "block"}


# =========================================================
#   Render page‑specific content                            
# =========================================================
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    State("plot_store", "data"),
)

def render_page(path, plot_data):
    if path == "/plot" and plot_data:
        fig = go.Figure(plot_data)
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
                dcc.Graph(id="tsne-3d-graph", figure=fig, config={"displayModeBar": False}, style={"height": "100vh"}),
                html.Div("Plot generation is completed", className="text-center mt-2"),
            ]
        )
    return ""


# =========================================================
#   Regenerate button → go home                             
# =========================================================
@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("regenerate_button", "n_clicks"),
    prevent_initial_call=True,
)

def regenerate(n_clicks):
    if n_clicks:
        return "/"
    return no_update


# =========================================================
#   Run                                                     
# =========================================================
if __name__ == "__main__":
    import webbrowser

    port = 8050
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)
    app.run(debug=False, port=port)
