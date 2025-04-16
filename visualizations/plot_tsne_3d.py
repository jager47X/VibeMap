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
    COLLECTION
)

# Set default Plotly template
pio.templates.default = "plotly_white"

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Instantiate the Dash App ---
# Using a custom assets folder ("../Data/assets") because the image resides in Data/assets in the project root.
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.FLATLY]
app = dash.Dash(
    __name__,
    assets_folder="../Data/assets",  # Updated relative path from visualizations/ folder to Data/assets folder.
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)
app.title = "Vibe Map"

# === Data Loading and Processing Functions ===

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

def compute_tsne(embeddings, max_itr_input, random_state=42):
    logger.info("Performing PCA...")
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    logger.info("Running t-SNE...")
    tsne = TSNE(n_components=3, perplexity=30, n_iter=max_itr_input, init='pca',
                random_state=random_state, verbose=2)
    return tsne.fit_transform(reduced)

def prepare_dataframe(data, max_itr_input=1000):
    embeddings = np.array([d["embeddings"] for d in data])
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
            "z": coord[2]
        }
        records.append(record)
    df = pd.DataFrame(records)
    df.dropna(subset=["timestamp"], inplace=True)
    df["time_bucket"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    logger.info("Prepared DataFrame with %d rows", len(df))
    return df

def build_plot(df):
    logger.info("Building animated 3D Plot...")
    df = df.sort_values("timestamp")
    unique_dates = sorted(df["time_bucket"].unique())
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
    steps = [{
        "args": [[frame["name"]], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
        "label": frame["name"],
        "method": "animate"
    } for frame in [all_time_frame] + all_frames]
    usernames = sorted(df["username"].unique())
    emotions = sorted(df["emotion"].unique())
    username_buttons = [{
        "label": "All Users",
        "method": "restyle",
        "args": [{
            "x": [df["x"]],
            "y": [df["y"]],
            "z": [df["z"]],
            "text": [df["title"]],
            "customdata": [np.stack([df["username"], df["timestamp"].astype(str), df["tweets"]], axis=-1)],
            "marker.color": [df["emotion"].map(EMOTION_COLOR_MAP)]
        }]
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
        "args": [{
            "x": [df["x"]],
            "y": [df["y"]],
            "z": [df["z"]],
            "text": [df["title"]],
            "customdata": [np.stack([df["username"], df["timestamp"].astype(str), df["tweets"]], axis=-1)],
            "marker.color": [df["emotion"].map(EMOTION_COLOR_MAP)]
        }]
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
    emotion_legend_annotations = [{
        "x": 1.15,
        "y": 0.9 - 0.05 * i,
        "xref": "paper",
        "yref": "paper",
        "text": f'<span style="color:{EMOTION_COLOR_MAP[e]};"><b>{e}</b></span>',
        "showarrow": False,
        "font": {"size": 12}
    } for i, e in enumerate(EMOTION_COLOR_MAP.keys())]
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
    fig = go.Figure(
        data=all_time_frame["data"],
        layout=layout,
        frames=[all_time_frame] + all_frames
    )
    return fig

# --- Global App Layout ---
app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="remaining_time_store"),
    dcc.Store(id="plot_store"),
    dcc.Interval(id="countdown-interval", interval=1000, disabled=True),
    # Persistent Controls Container
    html.Div(id="controls-container", children=[
        dbc.Row(
            dbc.Col(
                html.H1("Vibe Map", className="text-center my-4"),
                width=12
            )
        ),
        dbc.Row([
            dbc.Col([
                dbc.Label("Max Records (default 100):"),
                dbc.Input(id="limit_input", type="number", placeholder="Enter max records", min=1)
            ], width=6),
            dbc.Col([
                dbc.Label("Max t-SNE Iterations (min 250):"),
                dbc.Input(id="max_itr_input", type="number", placeholder="Enter max iterations", min=250, value=250)
            ], width=6)
        ], className="mb-3"),
        dbc.Row(
            dbc.Col(
                dbc.Button("Generate Plot", id="generate_button", color="primary", className="mb-3", n_clicks=0),
                width="auto",
                className="d-flex justify-content-center"
            )
        ),
        dcc.Loading(
            id="loading-spinner",
            type="default",
            children=html.Div([
                html.Div(id="estimated-time", className="text-center"),
                html.Div(id="loading-message", className="text-center mt-2")
            ])
        ),
        # Big image to fill the rest of the page
        html.Div(
            html.Img(
                src="/assets/logo.png",
                style={
                    "width": "100%",         # Fill the container width.
                    "height": "80vh",         # Set container height.
                    "objectFit": "contain",   # Ensure the entire image is visible.
                    "transform": "scale(0.9)",# Optional: scales the image down by 10%
                    "margin": "auto"          # Center the image if there’s extra space.
                }
            ),
            style={"marginTop": "1rem"}
        ),

    ]),
    # Page content (for the plot) appears here.
    html.Div(id="page-content")
], fluid=True)

# --- Callback: Update Estimated Time & Countdown ---
@app.callback(
    Output("estimated-time", "children"),
    Output("remaining_time_store", "data"),
    Output("countdown-interval", "disabled"),
    Output("loading-message", "children"),
    Input("limit_input", "value"),
    Input("max_itr_input", "value"),
    Input("generate_button", "n_clicks"),
    Input("countdown-interval", "n_intervals"),
    Input("url", "pathname"),
    State("remaining_time_store", "data")
)
def update_estimated_time(limit_value, max_itr_value, n_clicks, n_intervals, pathname, remaining):
    from dash import callback_context, no_update
    import math
    import dash_bootstrap_components as dbc
    from dash import html

    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Parse input values with fallback defaults.
    try:
        limit = int(limit_value) if limit_value is not None else 100
    except ValueError:
        limit = 100

    try:
        max_itr = int(max_itr_value) if max_itr_value is not None else 250
    except ValueError:
        max_itr = 250

    computed = math.ceil((limit / 10000) * (max_itr / 5))
    formatted_time = (
        f"{computed // 3600}h {computed % 3600 // 60}m {computed % 60}s"
        if computed >= 3600 else
        f"{computed // 60}m {computed % 60}s"
        if computed >= 60 else
        f"{computed}s"
    )

    spinner = dbc.Spinner(size="sm", color="primary", type="border")

    # When no countdown is active (for example, on first load or if no generate button clicked)
    if remaining is None:
        # Optionally, if you really want to check the pathname, you can do so here
        # but only return the default state if no countdown is active.
        return f"Estimated processing time: {formatted_time}", computed, True, ""

    # If the generate button was clicked, start or reinitialize the countdown.
    if "generate_button" in trigger:
        return (
            f"Estimated processing time: {computed} seconds",
            computed,
            False,  # Enable the countdown interval
            spinner
        )

    # If the user modifies input values (limit or max_itr) after the countdown has begun,
    # you might choose to ignore those changes until the current count is finished.
    if "limit_input" in trigger or "max_itr_input" in trigger:
        return no_update, no_update, no_update, no_update

    # Countdown: When the interval fires.
    if "countdown-interval" in trigger:
        # Ensure we have a valid starting point.
        if remaining is None:
            remaining = computed

        new_remaining = max(remaining - 1, 0)
        if new_remaining == 0:
            msg = dbc.Row(
                dbc.Col(html.Div(
                    "Estimation of the time may vary depending on your machine. Finalizing the plot...",
                    className="text-center"
                ), width=12)
            )
            return (
                f"Estimated processing time: 0 seconds",
                0,
                True,  # Disable the interval once finished.
                msg
            )
        else:
            return (
                f"Estimated processing time: {new_remaining} seconds",
                new_remaining,
                False,  # Continue running the interval.
                spinner
            )

    # Default behavior (if no relevant trigger is detected)
    return f"Estimated processing time: {formatted_time}", computed, True, ""


# --- Callback: Generate Plot and Redirect ---
@app.callback(
    Output("plot_store", "data"),
    Output("url", "pathname"),
    Input("generate_button", "n_clicks"),
    State("limit_input", "value"),
    State("max_itr_input", "value")
)
def generate_and_redirect(n_clicks, limit_value, max_itr_value):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    try:
        limit = int(limit_value) if limit_value is not None else 100
    except ValueError:
        limit = 100
    try:
        max_itr = int(max_itr_value)
        if max_itr < 250:
            max_itr = 250
            logger.info("Max iterations set to minimum value of 250.")
    except (ValueError, TypeError):
        max_itr = 250

    data = load_mongo_data(EMOTION_ASSIGNED_TWEETS_COLLECTION, limit)
    if not data:
        logger.warning("No data loaded.")
        return None, "/"
    df = prepare_dataframe(data, max_itr)
    fig = build_plot(df)
    return fig.to_plotly_json(), "/plot"

# --- Callback: Toggle Controls Visibility Based on URL ---
@app.callback(
    Output("controls-container", "style"),
    Input("url", "pathname")
)
def toggle_controls(pathname):
    if pathname == "/plot":
        return {"display": "none"}
    return {"display": "block"}

# --- Callback: Render Page Content Based on URL ---
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    State("plot_store", "data")
)
def render_page_content(pathname, plot_data):
    if pathname == "/plot" and plot_data is not None:
        fig = go.Figure(plot_data)
        return html.Div([
            dbc.Row(
                dbc.Col(
                    dbc.Button("Regenerate the Plot", id="regenerate_button", color="primary", n_clicks=0),
                    width={"size": 4, "offset": 4},
                    className="text-center"
                )
            ),
            dcc.Graph(
                id="tsne-3d-graph",
                figure=fig,
                config={"displayModeBar": False},
                style={"height": "100vh", "width": "100%"}
            ),
            html.Div("Plot generation is completed", style={"textAlign": "center", "marginTop": "1rem"})
        ])
    return ""

# --- Callback: Handle Regenerate Button on Plot Page ---
@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("regenerate_button", "n_clicks"),
    prevent_initial_call=True
)
def regenerate_plot(n_clicks):
    if n_clicks:
        return "/"
    return no_update

# --- Automatically Open Browser and Run ---
if __name__ == "__main__":
    import webbrowser
    port = 8050
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)
    app.run(debug=False, port=port)
