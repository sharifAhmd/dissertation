import warnings
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.autograd import Variable
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fill_type = "fractal"
seq_length = 120
labels_length = 30
n_features = 1


# Set page configuration 
st.set_page_config(
    page_title="Solar Flare Forecasting",
    page_icon="☀️",
    layout="wide"
)

# Add content to your Streamlit app
st.markdown("# Welcome to Solar-Flare Forecasting")


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
    df_xl["flux"] = df_xl["flux"]
    lst_flux = df_xl["flux"].tolist()
    lst_flux = lst_flux[-120:]
    latest_flux_val = lst_flux[-1:]
    lst_flux = [[-np.log(flux)] for flux in lst_flux]

    return df_xl, lst_flux, latest_flux_val

def predict_flare(max_value):
    """
    Map the assigned class
    """
    if max_value > 1e-05 and max_value <= 1e-04:
        return "M-class flare"
    elif max_value > 1e-04:
        return "X-class flare"
    return "No Flare"


def model_instance():
    """
    Create instance of trained LSTM model with
    """
    class Encoder(nn.Module):
        def __init__(self, seq_len, n_features, embedding_dim=32):
            super(Encoder, self).__init__()
            self.seq_len, self.n_features = seq_len, n_features
            self.embedding_dim, self.hidden_dim = embedding_dim, 16  
            self.num_layers = 1
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.35
            )
    
        def forward(self, x):
            device = x.device
            x = x.reshape((1, self.seq_len, self.n_features))
            h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device))
            c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device))
            x, (hidden, cell) = self.lstm(x, (h_1, c_1))
            return x, hidden, cell

    class Decoder(nn.Module):
        def __init__(self, seq_len, input_dim=32, n_features=1):
            super(Decoder, self).__init__()
            self.seq_len, self.input_dim = seq_len, input_dim
            self.hidden_dim, self.n_features = 16, n_features  
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=16,  
                num_layers=1,
                batch_first=True,
                dropout=0.35
            )
            self.output_layer = nn.Linear(self.hidden_dim, n_features)

        def forward(self, x, input_hidden, input_cell):
            device = x.device
            x = x.reshape((1, 1, 1))
            x, (hidden_n, cell_n) = self.lstm(x, (input_hidden, input_cell))
            x = self.output_layer(x)
            return x, hidden_n, cell_n

    class Seq2Seq(nn.Module):
        def __init__(self, seq_len, n_features, embedding_dim=32, output_length=labels_length): 
            super(Seq2Seq, self).__init__()
            self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
            self.output_length = output_length
            self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

        def forward(self, x, prev_y):
            encoder_output, hidden, cell = self.encoder(x)
            targets_ta = []
            prev_output = prev_y
            for out_days in range(self.output_length):
                prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell)
                hidden, cell = prev_hidden, prev_cell
                prev_output = prev_x
                targets_ta.append(prev_x.reshape(1))
            targets = torch.stack(targets_ta)
            return targets

    model = Seq2Seq(seq_length, n_features)
    model = model.to(device)
    model.load_state_dict(torch.load(f"Model/best_model_{fill_type}_seq_{seq_length}_label_{labels_length}.pt", map_location="cpu"))    

    return model

def ger_prediction():
    """
    To runn all the functions
    """
    model = model_instance()

    df_xl, lst_flux, latest_flux_val = get_realtime_inference_data()
    time_ = df_xl['time_tag'].iloc[0]
    input_ = Variable(torch.Tensor(lst_flux))
    
    with torch.no_grad(): 
        output_ = model(input_, input_[seq_length-1:seq_length,:])

    output_ = [np.exp(-v[0]) for v in output_.tolist()]
    return df_xl, output_


data = st.container()
count = st_autorefresh(interval=300000, limit=100000, key="solar_flare")
df, pred_ = ger_prediction()
df = df.tail(120)
df["pred_flux"] = np.nan

last_time = df['time_tag'].iloc[-1]
new_times = [last_time + pd.Timedelta(minutes=1*(i+1)) for i in range(len(pred_))]

# New DataFrame with the new times and flux values
new_data = {'time_tag': new_times, 'flux': np.nan, "pred_flux": pred_}
new_df = pd.DataFrame(new_data)


# Taking the last value of the original data to have a good visual representation
new_df['pred_flux'].iloc[0] = df['flux'].iloc[-1]

# Append the new DataFrame to the original DataFrame
df = pd.concat([df, new_df], ignore_index=True)
df = df[["time_tag","flux", "pred_flux"]]
# Set the "date" column as the index
load_df = df.set_index('time_tag')

flare_class = predict_flare(max(pred_))
# Display the line chart with dates on the x-axis
st.line_chart(load_df, y = ["flux", "pred_flux"], color=["#0000FF", "#FF0000"])
current_time = datetime.now()
formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
st.markdown(f"## There is a chance of having  {flare_class} in the next 30 minutes based on GOES-18 satellite XRSB data.")
print(f"##{current_time} There is a chance of having  {flare_class} in the next 30 minutes based on GOES-18 satellite XRSB data.")
