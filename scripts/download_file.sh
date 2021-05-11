##Download images for classification

current_dir = os.getcwd().replace("\\","/")
data_directory = current_dir+'/data'

try: ## vérifier si un dossier data existe déjà (sinon, le créer) et le prendre comme répertoire courant afin d'y faire les téléchargements
    os.makedirs(data_directory) ##former le dossier data
    os.chdir(data_directory)
except :
    os.chdir(data_directory) 

##le code précédent va créer un répertoire data dans le répertoire current_dir grâce au chemin data_directory. Si on ne veut pas cela on peut créer le dossier data dans le répertoire parent en procédant comme suit:
#parent_directory = current_dir.replace('/'+os.path.basename(current_dir),"") ##former le dossier parent
#os.chdir(parent_directory)
#data_directory = parent_directory+'/data'
##puis exécuter le bloc try ...

##téléchargement
!curl -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
!curl -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
!curl -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

##dézipper
from zipfile import ZipFile

with ZipFile('GTSRB_Final_Training_Images.zip', 'r') as dezip:
    dezip.extractall('GTSRB_Final_Training_Images')
    
with ZipFile('GTSRB_Final_Test_Images.zip', 'r') as dezip:
    dezip.extractall('GTSRB_Final_Test_Images')
    
with ZipFile('GTSRB_Final_Test_GT.zip', 'r') as dezip:
    dezip.extractall('GTSRB_Final_Test_GT')

os.chdir(current_dir) ##retouner au précedent repertoire courant


##Download images for detection

current_dir = os.getcwd().replace("\\","/")
data_directory = current_dir+'/data'

try: ## vérifier si un dossier data existe déjà (sinon, le créer) et le prendre comme répertoire courant afin d'y faire les téléchargements
    os.makedirs(data_directory) ##former le dossier data
    os.chdir(data_directory)
except :
    os.chdir(data_directory) 

##le code précédent va créer un répertoire data dans le répertoire current_dir grâce au chemin data_directory. Si on ne veut pas cela on peut créer le dossier data dans le répertoire parent en procédant comme suit:
#parent_directory = current_dir.replace('/'+os.path.basename(current_dir),"") ##former le dossier parent
#os.chdir(parent_directory)
#data_directory = parent_directory+'/data'
##puis exécuter le bloc try ...

##téléchargement
!curl -O https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip
!curl -O https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TestIJCNN2013.zip
!curl -O https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/gt.txt

##dézipper
from zipfile import ZipFile

with ZipFile('TrainIJCNN2013.zip', 'r') as dezip:
    dezip.extractall('TrainIJCNN2013')
    
with ZipFile('TestIJCNN2013.zip', 'r') as dezip:
    dezip.extractall('TestIJCNN2013')


os.chdir(current_dir) ##retouner au précedent repertoire courant