from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import pickle

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
df=pd.read_csv('C:/Users/bedis/Desktop/flickr/cbir/photo_metadata/resultat2.csv')



Limages=[]

with open('C:/Users/bedis/Desktop/pkl1/j.pkl', 'rb') as f:
   j = pickle.load(f)

print(j)

for i in range(j,j+12000):
    
    link="https://live.staticflickr.com/{x}/{y}_{z}.jpg".format(x=df.iloc[i]['flickr_server'],y=df.iloc[i]['id'],z=df.iloc[i]['flickr_secret'])
    Limages.append(link)
    j+=1
    

model = vgg16.VGG16(weights='imagenet', include_top=True)


def feature_extractor(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
    fc2_features = model_extractfeatures.predict(x)[0]
    
    
    return fc2_features/np.linalg.norm(fc2_features)
L1=[]

with open('C:/Users/bedis/Desktop/pkl1/Lindice.pkl', 'rb') as f:
   Lindice= pickle.load(f)
with open('C:/Users/bedis/Desktop/pkl1/i.pkl', 'rb') as f:
   i = pickle.load(f)

for path in Limages:
    try:
        i+=1
        response = requests.get(path)

        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        img = image.img_to_array(img)
        if img.shape==(224,224,3):
            L1.append(img)
            Lindice.append(i)
            
        
    except:
        
        pass

print(i)
L1=[feature_extractor(img) for img in L1]
with open('C:/Users/bedis/Desktop/pkl1/L.pkl', 'rb') as f:
   L = pickle.load(f)
L=L+L1



print(len(L))
with open('C:/Users/bedis/Desktop/pkl1/L.pkl', 'wb') as f:
  mynewlist = pickle.dump(L,f)
with open('C:/Users/bedis/Desktop//pkl1/j.pkl', 'wb') as f:
  mynewlist = pickle.dump(j,f)
with open('C:/Users/bedis/Desktop/pkl1/Lindice.pkl', 'wb') as f:
  mynewlist = pickle.dump(Lindice,f)
with open('C:/Users/bedis/Desktop//pkl1/i.pkl', 'wb') as f:
  mynewlist = pickle.dump(i,f)




