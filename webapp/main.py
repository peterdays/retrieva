import os
import time
from urllib.parse import urljoin

import dash_bootstrap_components as dbc
import diskcache
import httpx
from dash import (Dash, DiskcacheManager, Input, Output, State, callback, dcc,
                  html)
from dash.exceptions import PreventUpdate

# setting up the diskcache for background callbacks
diskcache_path = os.path.join("artifacts/cache")
if not os.path.exists(diskcache_path):
    os.makedirs(diskcache_path)
cache = diskcache.Cache(diskcache_path)
background_callback_manager = DiskcacheManager(cache)

app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA],
           background_callback_manager=background_callback_manager)

app.title = 'Retrieva'

app.layout = dbc.Row([
    dbc.Col(html.H2("Retrieva Demo 🚀"), width={"size": 2, "offset": 5}),
    dbc.Col(
        dcc.Textarea(
            id='markdown_input',
            placeholder="Ask me anything about our docs!",
            style={'width': '100%', 'height': 150}
        ),
        width={"size": 6, "offset": 3}
    ),
    dbc.Col(
        dbc.Row(dbc.Button('Go!', id='display_button', n_clicks=0,
                            style={"margin": "1em"}),
                justify="end"),
        width={"size": 1, "offset": 8}
    ),
    html.Br(),
    dbc.Col(
        dcc.Markdown(id='display_area'),
        width={"size": 6, "offset": 3}
    )
], style={"margin-top": "5em"})


@callback(
    Output('display_area', 'children'),
    [Input('display_button', 'n_clicks')],
    [State('markdown_input', 'value')],
    progress=Output("display_area", "children"),
    progress_default="",
    background=True,
    interval=100
)
def update_output(set_progress, n_clicks, user_prompt):
    if n_clicks > 0:
        url = urljoin(os.environ["RAG_API_URL"], "query")
        with httpx.stream('GET', url,
                          params={"query": user_prompt}) as r:
            sentence = ""
            for chunk in r.iter_raw():  # or, for line in r.iter_lines():
                time.sleep(0.01)  # if iteration too quick streaming wont happen
                sentence += chunk.decode("utf-8")
                set_progress(sentence)

        return sentence
    raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=False, port=4444)
