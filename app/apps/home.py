import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


layout = html.Div([
    html.Hr(),
    dbc.Toast(
        [html.P(" ", className="mb-0")],
        header="RECONNAISSANCE ET DETECTION DES PANNEAUX DE SIGNALISATION", 
        style={
            "text-align": "center", 
            "background-color": ""
            },
    ),
    dcc.Link("Aller à l'app de reconnaisance", href='/apps/recognition'),
    html.Br(),
    dcc.Link("Aller à l'app de détection", href='/apps/detection'),
    
])

