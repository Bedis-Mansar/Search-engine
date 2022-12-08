import time
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from PIL import Image
image = Image.open('supcom.png')
st.title("Search Engine Application")
with st.sidebar:
    st.image(image)
    st.markdown('### About the project')
    st.text('This project was conducted in regards\n to the rise of importance of \nsearch engines. The aim is to \nuse state-of-the-art \ntechnologies and methods to \noptimize their performance. \nAs a result, this application is \na fast and accurate content-and-\ntext based search engine.')
    with st.container():
        st.write("━━━━━━━━━━━━━━━━")
    with st.expander("Help"):
        st.text('This application functions\n similarly to Google\n search. You can search for your\n desired result by uploading an\n image or by typing a text.\nAdditionally, you can search\n using both image and text\n at the same time for even more\nprecise results.')
    with st.expander("Done By"):
        st.text("""
        Rihab JERBI
        Bedis MANSAR
                             INDP3 AIM
                             2022/2023
    """)
col1,col2 = st.columns([2,1])
# displays a file uploader widget
with col1 :
    st.subheader("Search by image")
    image = st.file_uploader("Choose an image 👇")
with col2 :
    st.subheader("Search by text")
    text=st.text_input("Enter a text 👇")

if st.button("Search"):
    if image is not None:
        if len(text)>0:
            headers = {'accept': 'application/json'}
            files = {"file": image.getvalue()}
            params = {'q': text}
            res = requests.post(f"http://127.0.0.1:8080/searchquery/imagetextsearch",headers=headers, params=params, files=files)
            results = res.json()
            for i in range(len(results['hits']['hits'])):
                result = results['hits']['hits'][i]
                res = {}
                res["url"] = result['_source']["imgUrl"]
                res["tags"]=result['_source']["tags"]
                image = Image.open(BytesIO(requests.get(res["url"]).content))
                st.image(image, width=500)
                st.write(res['tags'])
        else:
            headers = {'accept': 'application/json'}
            
            files = {"file": image.getvalue()}
            res = requests.post(f"http://127.0.0.1:8080/searchquery/imagesearch",headers=headers, files=files)
            results = res.json()
            for i in range(len(results['hits']['hits'])):
                result = results['hits']['hits'][i]
                res = {}
                res["url"] = result['_source']["imgUrl"]
                res["tags"]=result['_source']["tags"]
                image = Image.open(BytesIO(requests.get(res["url"]).content))
                st.image(image, width=500)
                st.write(res['tags'])           
    else:
        params = {'q': text}
        headers = {'accept': 'application/json'}
        response = requests.get(f"http://127.0.0.1:8080/searchquery/textsearch",headers=headers,params=params)
        results = response.json()
        for i in range(len(results['hits']['hits'])):
            result = results['hits']['hits'][i]
            res = {}
            res["url"] = result['_source']["imgUrl"]
            res["tags"]=result['_source']["tags"]
            image = Image.open(BytesIO(requests.get(res["url"]).content))
            st.image(image, width=500)
            st.write(res['tags'])
