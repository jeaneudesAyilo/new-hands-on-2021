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

from constants import CLASSES
from dash import no_update 
from joblib import load
from skimage.feature import hog 
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 

sys.path.insert(1, "C:/Users/jeane/Documents/new-hands-on-2021/notebooks/yolov3")
from configs import *
import yolov4
from yolov4 import Create_Yolo
from utils import load_yolo_weights, detect_image

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("C:/Users/jeane/Documents/new-hands-on-2021/notebooks/checkpoints/yolov3_custom")

pca = load("C:/Users/jeane/Documents/new-hands-on-2021/models/pca.joblib")
svm = load("C:/Users/jeane/Documents/new-hands-on-2021/models/new_svm.joblib")

##read yaml file
a_yaml_file = open("C:/Users/jeane/Documents/new-hands-on-2021/app/app.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
my_variables = parsed_yaml_file['variables']


# Load DNN model
clf_dnn = tf.keras.models.load_model('C:/Users/jeane/Documents/new-hands-on-2021/models/traffic_signs_2021-03-19_20-35-31.h5')


def classify_image(image,classifier=None, image_box=None,resize=True):
  """Classify image by model

  Parameters
  ----------
  content: image content
  model: tf/keras classifier
    
  Returns
  -------
  class id returned by model classifier
  """
  
  if resize:
      image = image.resize((my_variables['IMAGE_WIDTH'], my_variables['IMAGE_HEIGHT']), box=image_box)
                                            # box argument clips image to (x1, y1, x2, y2)
  else:
      pass

  if classifier == "cnn":
      images_list = []
      image = np.array(image)
      images_list.append(image)
      predictions = clf_dnn.predict(np.array(images_list))
      a = np.argmax(predictions)
      b = np.max(predictions)
      dico = {'id_class_pred':a,'prob_class_pred':b}
      df = pd.DataFrame({"Classes" :list(CLASSES.values()), "Probabilité" : predictions.tolist()[0]})
      fig = px.bar(df,x="Classes",y= "Probabilité", title="Probabilité de chaque classe", template='plotly_dark')
      return dico, fig
          
  elif classifier == "svm":
      Features = []
      image_flat = np.asarray(image).flatten().tolist()
      image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY) ##convert to grayscale 
      ret,thresh_image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
      descriptor = hog(thresh_image, orientations=8,pixels_per_cell=(4,4)).tolist() #hog features extraction
      image_flat.extend(descriptor)
      Features.append(image_flat)#hog features saving
      Features=np.array(Features)
      
      reduct = pca.transform(Features)
      a = svm.predict(reduct).item() ## predicted class
      decision = svm.decision_function(reduct) # decision is a voting function
      predictions = np.exp(decision)/np.sum(np.exp(decision),axis=1, keepdims=True) # softmax after the voting
      b = np.max(predictions)
    
      dico = {'id_class_pred':a,'prob_class_pred':b}
      df = pd.DataFrame({"Classes" :list(CLASSES.values()), "Probabilité" : predictions.tolist()[0]})
      fig = px.bar(df,x="Classes",y= "Probabilité", title="Probabilité de chaque classe", template='plotly_dark')
      return dico, fig
  
  else:
        pass

def pertub_image(image, n_pixel, seed = None):
    """Create a pertubation in the image by changing a given number of pixels value at random positions.
    
    Parameters
    ----------
    image: numpy array of the original image
    
    n_pixel: number of pixels to modify
    
    seed : random seed
    
    Returns
    -------
    A numpy array that represent the per
    """
    if seed != None:
        np.random.seed(seed)
    
    pixels = np.random.randint(low =30,size = (n_pixel, 2))
    
    rgb = np.random.randint(low =256,size = (n_pixel, 3))
    
    image2 = np.copy(image)
    for i in range(n_pixel):
        image2[pixels[i,0],pixels[i,1],:] = np.array([rgb[i,0],rgb[i,1],rgb[i,2]])
       
    return image2    

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

pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}

##define a dropdown list to choose the model
dropdown1 = dcc.Dropdown(
    id='choix_model',
    options=[
        {'label': 'CNN', 'value': 'cnn'},
        {'label': 'SVM', 'value': 'svm'}
    ],
    value=None,
    multi=False,style = {'width':"40%"})

##
radio = dcc.RadioItems(
    id='radio_item_change_pixels',
    options=[
        {'label': 'OUI', 'value': 'oui'},
        {'label': 'NON', 'value': 'non'}
    ],
    value='non'
)

##define a slider to select the number of pixel to be modified
slider = dcc.Slider(
    id = "choisir_nb_pixels",
    min=0,
    max=900,##dû au fait qu'on aura des images 30*30
    step=10,
    value=0
    ) 

# Define application layout

layout = html.Div([
    html.Hr(),
    dbc.Toast(
        [html.P(" ", className="mb-0")],
        header="RECONNAISSANCE DES PANNEAUX DE SIGNALISATION", 
        style={
            "text-align": "center", 
            "background-color": ""
            },
    ),
      
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Cliquer-déposer ou ',
                    html.A('sélectionner une image')
        ],style={'textAlign': 'center'}),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin' : '10px 10px 50px 300px'
        }
    ),
    dcc.Markdown('Choisissez le modèle à utiliser'),
    html.Div(className='model', children=[dropdown1]),
    html.Button('Submit', id='button'),
    html.Div(id='mon-image',children =[]),
    html.Div([dcc.Store(id='element-to-hide')]),
    dcc.Markdown('''Voulez-vous tester une détérioration de l'image ?'''),
    html.Div(id = "image_modifiée", children=[radio]),
    html.Div(id = "mon-slider",children = [html.Br(),dcc.Slider(
        id = "choisir_nb_pixels",
        min=0,
        max=900,##dû au fait qu'on aura des images 30*30
        step=10,
        marks={i: '{} pixels'.format(i) for i in range(0,901,100)},    
        value=0
        ),html.Br(),
        dcc.Dropdown(
        id='choix_model2',
        options=[
            {'label': 'CNN', 'value': 'cnn'},
            {'label': 'SVM', 'value': 'svm'}
        ],
        value="cnn",
        multi=False,style = {'width':"40%"})                
    ],style = {"display":"none"}),
    html.Br(),
    html.Div(id='modification',children =[]),
    html.Div(id='detection',children =[]),
    dcc.Link("Aller à l'app de détection", href='/apps/detection'),html.Br(),
    dcc.Link("Accueil", href='/apps/home')
])


@app.callback([Output('mon-image', 'children'),
               Output('element-to-hide', 'data'),
               Output('mon-slider', 'style'),
               Output('radio_item_change_pixels', 'value')],
              [Input('bouton-chargement', 'contents')],
              [Input(component_id='button', component_property='n_clicks')],
              [State('choix_model', component_property='value')])
              

def update_output(contents,n_clicks,value_model):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            if value_model !=None: ## initialement, si aucun model n'est choisi, ne rien faire
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                clf_predictions = None ;graph = None
                clf_predictions,graph = classify_image(image,classifier = value_model,resize=False)
                return (html.Div([
                    html.Hr(),
                    html.Img(src=contents),
                    html.H3('Classe prédite : {}'.format(CLASSES[clf_predictions['id_class_pred']])),
                    html.H3('Probabilité de la classe prédite : {:0.3f}'.format(clf_predictions['prob_class_pred'])),
                    html.Div([html.Hr(),dcc.Graph(id='barchart',figure=graph)]),
                    html.Hr(),
                    #html.Div('Raw Content'),
                    #html.Pre(contents, style=pre_style)
                ]), contents,{"display":"none"},'non') 
            elif value_model ==None:
                return (no_update,no_update,{"display":"none"},'non')
        else:
            try:
                if value_model !=None: ## initialement, si aucun model n'est choisi, ne rien faire
                    # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                    # fichier base 64 --> image PIL
                    image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                    # image PIL --> conversion PNG --> buffer mémoire 
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    # buffer mémoire --> image base 64
                    buffer.seek(0)
                    img_bytes = buffer.read()
                    content_string = base64.b64encode(img_bytes).decode('ascii')
                    # Appel du modèle de classification
                    clf_predictions = None ;graph = None
                    clf_predictions,graph = classify_image(image,classifier = value_model) 
                    # Affichage de l'image
                    return (html.Div([
                        html.Hr(),
                        html.Img(src='data:image/png;base64,' + content_string),
                        html.H3('Classe prédite : {}'.format(CLASSES[clf_predictions['id_class_pred']])),
                        html.H3('Probabilité de la classe prédite : {:0.3f}'.format(clf_predictions['prob_class_pred'])),
                        html.Div([html.Hr(),dcc.Graph(id='barchart',figure=graph)]),
                        html.Hr(),                        
                    ]), contents,{"display":"none"},'non')
                elif value_model ==None:
                    return (no_update, no_update,{"display":"none"},'non')
            except:
                return (html.Div([
                    html.Hr(),
                    html.Div('Uniquement des images svp : {}'.format(content_type)),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ]),no_update,{"display":"none"},'non')
    return (no_update, no_update,{"display":"none"},'non')

@app.callback(Output('mon-slider', 'style'),
      [Input(component_id = "radio_item_change_pixels", component_property='value')])


def set_slider(value):
    if value== "non":
        return {"display":"none"}
    else :
        return {"display":"block"}
    
    
@app.callback(Output('modification', 'children'),
      [Input("mon-slider",'style'),      
      Input("choisir_nb_pixels", 'value'),
      Input("choix_model2", 'value'),
      State('element-to-hide', 'data')
      ])

def display_pertub_image(style,value_pixel,value_model,value_contents):
    if (value_contents !=None and style == {"display":"block"}):
        content_type, content_string = value_contents.split(',')
        image = Image.open(io.BytesIO(base64.b64decode(content_string)))
        image = image.resize((my_variables['IMAGE_WIDTH'], my_variables['IMAGE_HEIGHT']))
        image = np.array(image)
        modified_array = pertub_image(image, value_pixel, seed = None)
        fig, ax = plt.subplots(1,1)
        ax.imshow(modified_array)
        ax.axis('off')
        out_url = fig_to_uri(fig)
        #clf_predictions = None 
        new_clf_predictions,_ = classify_image(modified_array,classifier = value_model,resize=False)
        return html.Div([
            html.Hr(),
            html.Img(src = out_url),
            html.H3('Classe prédite : {}'.format(CLASSES[new_clf_predictions['id_class_pred']])),
            html.H3('Probabilité de la classe prédite : {:0.3f}'.format(new_clf_predictions['prob_class_pred'])),
            html.Hr(),
            #html.Div('Raw Content'),
            #html.Pre(contents, style=pre_style)
        ])
    else:
        return #no_update #dcc.Markdown('''Veuillez charger une image ''')
    

