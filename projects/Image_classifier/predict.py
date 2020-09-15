import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
import argparse as ap
import logging
import warnings
from PIL import Image

warnings.filterwarnings("ignore")
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = ap.ArgumentParser(description = "image classification parser")
parser.add_argument("image_path",action="store")
parser.add_argument("saved_model",action="store")
parser.add_argument("--top_k",action="store",default=5,dest="top_k",required=False,type=int)
parser.add_argument("--category_names",action="store",dest="category_names")

results = parser.parse_args()

image_path = results.image_path
saved_model = results.saved_model
cat_file = results.category_names

if results.top_k==None:
    top_k = 5
else:
    top_k = results.top_k
    

def process_image(image):
    size = 224
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,(size,size))
    image = image/255
    return image.numpy()

def predict(image_path,model,top_k=5):
    img = Image.open(image_path)
    test_image = np.asarray(img)
    test_image = process_image(test_image)
    final_img = np.expand_dims(test_image,axis=0)
    probab_predicts = model.predict(final_img)
    final_probs= -np.partition(-probab_predicts[0],top_k)[:top_k]
    final_classes= np.argpartition(-probab_predicts[0],top_k)[:top_k]
    return final_probs,final_classes


model = tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer':hub.KerasLayer})
img = np.asarray(Image.open(image_path)).squeeze()[0]
probs,classes = predict(image_path,model,top_k)

if cat_file == None:
    with open('label_map.json','r') as  file:
        class_names = json.load(file)
    keys = [str(p+1) for p in list(classes)]
    classes = [class_names.get(k) for k in keys]
else:
    with open(cat_file,'r') as  file:
        class_names = json.load(file)
    keys = [str(p+1) for p in list(classes)]
    classes = [class_names.get(k) for k in keys]

print("top {} class probabilities for image are".format(top_k))

for i in np.arange(top_k):
    print("class: {}".format(classes[i]))
    print("probability: {:.3%}".format(probs[i]))
    print("\n")
