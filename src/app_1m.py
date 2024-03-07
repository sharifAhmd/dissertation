#Load libraries needed
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timedelta
import torch
import torch.nn as nn

import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Set page configuration 
st.set_page_config(
    page_title="Welcome To Solar FLare prediction",
    page_icon="☀️",
    layout="wide"
)

# Add content to your Streamlit app
st.markdown("# Welcome to Solar-Flare prediction")


def get_realtime_inference_data():
    """
    Get real time data from NOAA website
    """
    url = 'https://services.swpc.noaa.gov/json/goes/secondary/xrays-6-hour.json'

    ## Send a GET request to the URL
    response = requests.get(url)
    data = []
    ## Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON content
        data = response.json()
    df = pd.DataFrame(data)
    df_xl = df.iloc[lambda x: x.index % 2 == 1].reset_index(drop=True)
    df_xl = df_xl.replace(0.0, np.nan)
    df_xl = df_xl.ffill()

    df_xl['time_tag'] = pd.to_datetime(df_xl['time_tag'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize('UTC')

    # Convert the 'time_tag' from UTC to UK time zone (Europe/London)
    df_xl['time_tag'] = df_xl['time_tag'].dt.tz_convert('Europe/London')
    
    lst_flux = df_xl["flux"].tolist()
    lst_flux = lst_flux[-180:]
    latest_flux_val = lst_flux[-1:]
    lst_flux = [-np.log(flux) for flux in lst_flux]

    return df_xl, lst_flux, latest_flux_val

def predict_flare(values):
    """
    Map the assigned class
    """
    count_above_threshold = sum(value > 1e-05 for value in values)
    
    if count_above_threshold >= 2:
        max_value = min(values)
        if max_value > 1e-05 and max_value <= 1e-04:
            return "M-class flare"
        elif max_value > 1e-04:
            return "X-class flare"
    return "No Flare"


def model_instance(window_size):
    """
    Create instance of trained LSTM model with given window size
    """
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=window_size, hidden_size=200, num_layers=1, batch_first=True)
            self.linear = nn.Linear(200, 1)
        
        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x
    
    model = LSTMModel()
    model.load_state_dict(torch.load(f"Model/best_model_xl_{window_size}_minutes.pth"))
    model.eval()
    
    return model

def ger_prediction():
    """
    To runn all the functions
    """
    model_60 = model_instance(60)
    model_120 = model_instance(120)
    model_180 = model_instance(180)


    df_xl, lst_flux, latest_flux_val = get_realtime_inference_data()
    time_ = df_xl['time_tag'].iloc[0]
    input_60 = torch.tensor(lst_flux[-60:], dtype=torch.float).unsqueeze(0)
    input_120 = torch.tensor(lst_flux[-120:], dtype=torch.float).unsqueeze(0)
    input_180 = torch.tensor(lst_flux[-180:], dtype=torch.float).unsqueeze(0)

    with torch.no_grad(): 
        output_60 = model_60(input_60)
        output_120 = model_120(input_120) 
        output_180 = model_180(input_180)

    pred_60 = np.exp(-output_60.tolist()[0][0])
    pred_120 = np.exp(-output_120.tolist()[0][0])
    pred_180 = np.exp(-output_180.tolist()[0][0])
    fluxes = [pred_60, pred_120, pred_180]

    pred_flare = predict_flare(fluxes)
    return df_xl, latest_flux_val, time_, pred_60, pred_120, pred_180, pred_flare


data = st.container()
count = st_autorefresh(interval=59000, limit=100, key="solar_flare")
df, latest_flux_val, time_, pred_60, pred_120, pred_180, pred_flare = ger_prediction()


# Set the "date" column as the index
load_df = df.set_index('time_tag')

# Display the line chart with dates on the x-axis
st.line_chart(load_df["flux"])
current_time = datetime.now()
formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
st.markdown(f"## There will be {pred_flare} in the next upcoming minute.")
st.markdown(f"## Predicted flux from based of the last 60 minutes is: <span style='color: green;'>{pred_60} </span>", unsafe_allow_html=True)
st.markdown(f"## Predicted flux from based of the last 120 minutes is: <span style='color: green;'>{pred_120} </span>", unsafe_allow_html=True)
st.markdown(f"## Predicted flux from based of the last 180 minutes is: <span style='color: green;'>{pred_180} </span>", unsafe_allow_html=True)
