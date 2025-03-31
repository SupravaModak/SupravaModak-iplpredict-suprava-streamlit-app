# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 19:38:34 2025

@author: Suprava Modak
"""

import streamlit as st
import pickle
import pandas as pd
image_path = "iplimage.jpg"

teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Lucknow Super Giants',
 'Gujarat Titans']
cities= ['Hyderabad', 'Rajkot', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
       'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
       'Port Elizabeth', 'Durban', 'Centurion', 'East London',
       'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
       'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
       'Bengaluru']
pipe=pickle.load(open('pipe.pkl','rb'))
st.set_page_config(layout="wide")

# Function to load image as base64
import base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
bg_image = get_base64_of_image("iplimage.jpg")  # If it's in the same folder as prediction.py
# Custom CSS to set the background
def set_bg(image_file):
    bg_style = f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{image_file}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Apply background
bg_image = get_base64_of_image("D:\predictor\iplimage.jpg")  # Change to your image filename
set_bg(bg_image)
st.title('IPL Winner Predictor')
col1,col2=st.columns(2)
with col1:
    batting_team=st.selectbox('Select the Batting Team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select the Bowling Team',sorted(teams))

selected_city=st.selectbox('Select host city',sorted(cities))
target=st.number_input('Target')
col3,col4,col5= st.columns(3)
with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs completed')
with col5:
    wickets=st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs
    rrr=(runs_left*6)/balls_left
    
    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team + "-" +str(round(win*100)) + "%")
    st.header(bowling_team + "-" +str(round(loss*100)) + "%")
    
