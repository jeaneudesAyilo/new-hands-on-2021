import base64
import cv2
import io
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import sys
import tensorflow as tf
import yaml 

from dash.dependencies import Input,State, Output
from PIL import Image

sys.path.insert(1, "C:/Users/jeane/Documents/new-hands-on-2021/notebooks/yolov3")
from configs import *
import yolov4
from yolov4 import Create_Yolo
from utils import load_yolo_weights, detect_image

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("C:/Users/jeane/Documents/new-hands-on-2021/notebooks/checkpoints/yolov3_custom")

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = io.BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")#
    return "data:image/png;base64,{}".format(encoded)

from app import app

layout = html.Div([
    html.Hr(),
    dbc.Toast(
        [html.P(" ", className="mb-0")],
        header="DETECTION DES PANNEAUX DE SIGNALISATION", 
        style={
            "text-align": "center", 
            "background-color": ""
            },
    ),
      
    dcc.Upload(
    id='bouton-chargement_2',
    children=html.Div([
        'Cliquer-déposer ou ',
                html.A('sélectionner une image')
    ]),
    style={
        'width': '50%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'margin' : '10px 10px 50px 300px'
    }
  ),
  html.Div(id='detection',children =[]),
    dcc.Link("Aller à l'app de reconnaisance", href='/apps/recognition'),html.Br(),
    dcc.Link("Accueil", href='/apps/home')
])


@app.callback(Output('detection', 'children'),
      [Input("bouton-chargement_2",'contents'),
      ])

def display_detetion(contents):
    if contents!=None:
        content_type, content_string = contents.split(',')
        image = detect_image(yolo, image_path = content_string,type_image_path = "base64_string", output_path = "", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1,1,figsize=(10,6))
        ax.imshow(image)
        ax.axis('off')
        plt.imshow(image)
        out_url = fig_to_uri(fig)
        
        return html.Div([
            html.Hr(),
            html.Img(src = out_url),dcc.Markdown('''**prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]**

0: 'Speed limit (20km/h)' ; 
1: 'Speed limit (30km/h)' ;
2: 'Speed limit (50km/h)' ;
3: 'Speed limit (60km/h)' ;
4: 'Speed limit (70km/h)' ;
5: 'Speed limit (80km/h)' ;
7: 'Speed limit (100km/h)' ;
8: 'Speed limit (120km/h)' ;
9: 'No passing' ;
10: 'No passing veh over 3.5 tons' ;
15: 'No vehicles' ;
16: 'Veh > 3.5 tons prohibited' 

**mandatory = [33, 34, 35, 36, 37, 38, 39, 40]**

33: 'Turn right ahead' ;
34: 'Turn left ahead' ;
35: 'Ahead only' ;
36: 'Go straight or right' ;
37: 'Go straight or left' ;
38: 'Keep right' ;
39: 'Keep left' ;
40: 'Roundabout mandatory' 

**danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]**

11: 'Right-of-way at intersection' ;
18: 'General caution' ;
19: 'Dangerous curve left' ;
20: 'Dangerous curve right' ;
21: 'Double curve' ;
22: 'Bumpy road' ;
23: 'Slippery road' ;
24: 'Road narrows on the right' ;
25: 'Road work' ;
26: 'Traffic signals' ;
27: 'Pedestrians' ;
28: 'Children crossing' ;
29: 'Bicycles crossing' ;
30: 'Beware of ice/snow' ;
31: 'Wild animals crossing' 

**other = [6,12,13,14,17,32,41,42]**

6: 'End of speed limit (80km/h)' ;
12: 'Priority road' ;
13: 'Yield' ;
14: 'Stop' ;
17: 'No entry' ;
32: 'End speed + passing limits' ;
41: 'End of no passing' ;
42: 'End no passing veh > 3.5 tons' 

                                                 ''')])
    
    else :
        return  
    