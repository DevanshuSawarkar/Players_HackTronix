import streamlit as st
import numpy as np
from PIL import Image
from utils import *

labels = ["Organic", "Recyclable", "Non-Recyclable"]

st.markdown("""
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d73">Smart </span>Garbage</h1></center>
    <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
  </div>
""", unsafe_allow_html=True)

st.markdown("<center><h3 style='color: #008080; margin-top: -20px'>Check the type here</h3></center>", unsafe_allow_html=True)

opt = st.selectbox("Upload Image\n", ('Please Select', 'Upload image from device'))

if opt == 'Upload image from device':
    file = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)
        st.image(image, width=300, caption='Uploaded Image')
        
        if st.button('Predict'):
            img = preprocess(image)
            model = model_arc()
            model.load_weights("./models/modelnew.h5")
            
            prediction = model.predict(img[np.newaxis, ...])
            category = labels[np.argmax(prediction)]  # Selects the highest probability class
            
            st.info(f'Hey! The uploaded image has been classified as "{category}"')
