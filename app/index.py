import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import home, app_recognition, app_detection


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/apps/recognition':
        return app_recognition.layout
    elif pathname == '/apps/detection':
        return app_detection.layout
    else: 
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)