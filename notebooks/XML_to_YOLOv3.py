#================================================================
#   Adapted from: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description :used to convert XML labels to YOLOv3 training labels
#
#================================================================
import xml.etree.ElementTree as ET
import os
import glob

foldername = os.path.basename(os.getcwd())
if foldername == "tools": os.chdir("..")


data_dir = "./data/TrainIJCNN2013/general/Dataset/"
Dataset_names_path = "./data/TrainIJCNN2013/Dataset_names.txt"
Dataset_train = "./data/TrainIJCNN2013/Dataset_train.txt"
Dataset_val = "./data/TrainIJCNN2013/Dataset_val.txt"
is_subfolder = False

Dataset_names = []
      
def ParseXML(img_folder, file):
    for xml_file in glob.glob(img_folder+'/*.xml'):
        tree=ET.parse(open(xml_file))
        root = tree.getroot()
        image_name = root.find('filename').text
        img_path = img_folder+'/'+image_name
        for i, obj in enumerate(root.iter('object')):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in Dataset_names:
                Dataset_names.append(cls)
            cls_id = Dataset_names.index(cls)
            xmlbox = obj.find('bndbox')
            OBJECT = (str(int(float(xmlbox.find('xmin').text)))+','
                      +str(int(float(xmlbox.find('ymin').text)))+','
                      +str(int(float(xmlbox.find('xmax').text)))+','
                      +str(int(float(xmlbox.find('ymax').text)))+','
                      +str(cls_id))
            img_path += ' '+OBJECT
        print(img_path)
        file.write(img_path+'\n')

def run_XML_to_YOLOv3():
    for i, folder in enumerate(['train_set','val_set']):
        with open([Dataset_train,Dataset_val][i], "w") as file:
            print(os.getcwd()+data_dir+folder)
            img_path = os.path.join(os.getcwd()+data_dir+folder)
            if is_subfolder:
                for directory in os.listdir(img_path):
                    xml_path = os.path.join(img_path, directory)
                    ParseXML(xml_path, file)
            else:
                ParseXML(img_path, file)

    print("Dataset_names:", Dataset_names)
    with open(Dataset_names_path, "w") as file:
        for name in Dataset_names:
            file.write(str(name)+'\n')

run_XML_to_YOLOv3()
