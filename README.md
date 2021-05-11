# hands-on-2021
Hands-on project session 2021

* Build two models  (CNN and SVM) to classify traffic signs based on GTSRB datatset, and a build a model for traffic signs detection using pre-trained YOLOv3 model and GTSDB datatset.
* Build a small dash app to classify and detect images. In the classification part, we test also how some changes to the pixels values could affect the prediction. It is worth nothing that since the pixels values and their position were changed randomly, there is not a monotonic effect on the probability of the true class. An improvement would be to implement a one pixel attack. See : https://arxiv.org/pdf/1710.08864.pdf

## Installation

* clone this repository : `git clone https://github.com/jeaneudesAyilo/new-hands-on-2021.git`
* install the required packages : `pip install -r requirements.txt`
* downlod images with "scripts/downlaod_images.sh"
* Download yolov3 weights were from : https://pjreddie.com/media/files/yolov3.weights and put it in model directory
* One could get the weights of my yolo model on gtsdb from here : https://drive.google.com/file/d/1qJfrEgD1qnpnp1gzpaPFMO3B9ygcqCDT/view?usp=sharing  and put it in notebooks/checkpoints
* Run the dash app from your terminal, you might need to be in the `notebooks` directory to see it works, for example : 

`cd Documents/new-hands-on-2021/notebooks`

`python ../app/index.py` 

## Acknowledgments

* The yolo part is mainly based on the work of pythonlessons : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 ; https://www.youtube.com/watch?v=c-MVqtfHyAU&list=PLbMO9c_jUD473OgrKYYMLRMEz-XZjG--n
* Also take a look at  https://www.youtube.com/watch?v=10joRJt39Ns
* For dash tutorial, see https://www.youtube.com/channel/UCqBFsuAz41sqWcFjZkqmJqQ

## References 
- Dataset introduction 
- Images for classification: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
- Images for detection : https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html

## Demo
see a demo of the app : https://drive.google.com/file/d/11JahwCAeQQkbqhDtQVgk5S7EyLzz-kep/view?usp=sharing



