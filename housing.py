import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# House Price Prediction using PAS
This application will try to do prediction of housing price based on its certain feature variables
""")
st.write('---')
#-------------------------------------------------------------------------------------------------------
data = pd.read_csv("./data/housingx.csv")
#data = data.drop('id', axis=1)
st.header("Dataset Sample")
st.write(data.head())
#-------------------------------------------------------------------------------------------------------
st.header("Dataset Summary")
st.write(data.shape)
st.write(data.describe())
st.write('---')

# Specify variables feature and target------------------------------------------------------------------
X = data[['area','bedrooms','bathrooms','stories','guestroom','basement','hotwater','aircond','parking','prefarea','furnish']].values
Y = data[['price']].values

# Slide of Specify Input Parameters --------------------------------------------------------------------
st.sidebar.header('Specify Input: ')
def user_input_features():
    area        = st.sidebar.slider('Property Area in SqM', int(data['area'].min()), int(data['area'].max()))
    bedrooms    = st.sidebar.slider('No. of bedrooms', int(data['bedrooms'].min()), int(data['bedrooms'].max()))
    bathrooms   = st.sidebar.slider('No. of bathrooms', int(data['bathrooms'].min()), int(data['bathrooms'].max()))
    stories     = st.sidebar.slider('Level of Storeys', int(data['stories'].min()), int(data['stories'].max()))
    guestroom   = st.sidebar.slider('Has Guestroom', int(data['guestroom'].min()), int(data['guestroom'].max()))
    basement    = st.sidebar.slider('Has Basement', int(data['basement'].min()), int(data['basement'].max()))
    hotwater    = st.sidebar.slider('Has Hotwater', int(data['hotwater'].min()), int(data['hotwater'].max()))
    aircond     = st.sidebar.slider('Has Aircond', int(data['aircond'].min()), int(data['aircond'].max()))
    parking     = st.sidebar.slider('No. of parking', int(data['parking'].min()), int(data['parking'].max()))
    prefarea    = st.sidebar.slider('Is Prefered Area', int(data['prefarea'].min()), int(data['prefarea'].max()))
    furnish     = st.sidebar.slider('Is Furnish', int(data['furnish'].min()), int(data['furnish'].max()))

    datax = {'area':area, 'bedrooms': bedrooms,'bathrooms': bathrooms,'stories': stories, 'guestroom': guestroom,
            'basement': basement,'hotwater': hotwater,'aircond': aircond,'parking': parking, 'prefarea':prefarea, 'furnish':furnish}
    features = pd.DataFrame(datax, index=[0])
    return features
df = user_input_features()

# Show selected specified input parameters--------------------------------------------------------------------
st.header('Input parameters')
st.write(df)

# Build Regression Model -------------------------------------------------------------------------------------
model = RandomForestRegressor()
model.fit(X, Y)
# Prediction -------------------------------------------------------------------------------------------------
prediction = int(model.predict(df))
st.header('House price prediction: ')
st.write("Predicted house price based on selected parameter is: ", prediction)
st.write("Model accuracy score :", round(model.score(X, Y),4))
st.write('---')
