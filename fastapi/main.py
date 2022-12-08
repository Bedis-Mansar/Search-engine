from fastapi import FastAPI,UploadFile, File
from typing import Dict, List, Optional
from pydantic import BaseModel
import uvicorn
from PIL import Image
from elasticsearch import Elasticsearch
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from io import BytesIO
import pickle
import cv2
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
#######################################################
model = vgg16.VGG16(weights='imagenet', include_top=True)
def feature_extractor(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    
    model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
    fc2_features = model_extractfeatures.predict(x)[0]
    
    
    return fc2_features/np.linalg.norm(fc2_features)
#####################################################
with open('C:/Users/bedis/Desktop/pkl1/PCA795.pkl', 'rb') as f:
   PCA= pickle.load(f)
###########################################
index= 'indexproject'
source_no_vecs = ['imageId','tags','imgUrl']
es = Elasticsearch('127.0.0.1', port=9200)
##############################################################

# 1. Define an API object
app = FastAPI()


# 2. Define data type
class Searchengine(BaseModel):
    def searchtext(self,query):
        
        body = {
        "query":{"multi_match": {
          "query": query,
          "fields": ["tags"]
        }
      }
    }
        return es.search(index=index, body=body, size=20, _source=source_no_vecs)
    def imagesearch(self,img):
               
        img = image.img_to_array(img)
        feature=feature_extractor(img)
       
        feature=PCA.transform(feature.reshape(1,-1))[0].tolist()
        body = {
            "query":{
            "elastiknn_nearest_neighbors": {
              "field": "featurevector",
              "vec": {
                "values": feature
              },
              "model": "lsh",
              "similarity": "l2",
              "candidates": 100
            }
          }
        }
        return es.search(index=index, body=body, size=20,_source=source_no_vecs)
    def imagetextsearch(self,q,img):
              
        img = image.img_to_array(img)
        feature=feature_extractor(img)
       
        feature=PCA.transform(feature.reshape(1,-1))[0].tolist()
        body={"query": {
            "function_score": {
              "query": {
                "bool": {
                  "filter": {
                    "exists": {
                      "field": "featurevector"
                    }
                  },
                  "must": {
                    "multi_match": {
                      "query": q,
                      "fields": ["tags"]
                    }
                  }
                }
              },
              "boost_mode": "replace",
              "functions": [{
                "elastiknn_nearest_neighbors": {
                  "field": "featurevector",
                  "similarity": "l2",
                  "model": "lsh",
                  "candidates": 100,
                  "vec": {
                    "values": feature
                  }
                },
                "weight": 2
              }]
            }
          }
             }
        return es.search(index=index, body=body, size=20,_source=source_no_vecs)
ess =Searchengine()

# 3. Map HTTP method and path to python function
@app.get("/")
async def root():
    return {"message": "Hello Bedis !"}


@app.get("/searchquery")
async def function_demo_get():
    return {"message": "give a query input"}
#@app.get("/searchquery/{query}")
#async def searchquery(query:str):
    #return ess.searchtext(query)
@app.post("/searchquery/imagesearch")
async def searchquery( file: UploadFile = File([])):
    request_object_content = await file.read()

    image=np.array(Image.open(BytesIO(request_object_content)).resize((224,224)))

    return ess.imagesearch(image)
@app.get("/searchquery/textsearch")
async def textquery(q):
    return ess.searchtext(str(q))

@app.post("/searchquery/imagetextsearch")
async def searchcombine( q, file: UploadFile = File([])):
    request_object_content = await file.read()

    image=np.array(Image.open(BytesIO(request_object_content)).resize((224,224)))

    return ess.imagetextsearch(q,image)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080)