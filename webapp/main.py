import os

import dash_bootstrap_components as dbc
import httpx
from dash import Dash, Input, Output, State, callback, dcc, html

app = Dash(__name__)

app.layout = dbc.Col([
    html.H1("Retrieva Demo"),
    dcc.Textarea(
        id='markdown_input',
        placeholder="Ask me anything about our docs!",
        style={'width': '100%', 'height': 150}
    ),
    html.Button('Go!', id='display_button', n_clicks=0),
    html.Div(id='display_area')
])

@callback(
    Output('display_area', 'children'),
    [Input('display_button', 'n_clicks')],
    [State('markdown_input', 'value')]
)
def update_output(n_clicks, user_prompt):
    if n_clicks > 0:
        with httpx.stream('GET', os.environ["RAG_API_URL"],
                          params={"query": user_prompt}) as r:
            sentence = ""
            for chunk in r.iter_raw():  # or, for line in r.iter_lines():
                sentence += chunk.decode("utf-8")

        return sentence

if __name__ == '__main__':
    app.run_server(debug=True, port=4444)
