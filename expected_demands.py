import numpy as np
import wntr
from wntr.metrics.hydraulic import expected_demand
import pandas as pd
import torch
import torch_geometric
import networkx as nx
from neuralprophet import NeuralProphet
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import Data
import sys
import pytorch_lightning as pl


pl.seed_everything(0)

# load data
demands_2018 = pd.read_csv('dataset/2018_SCADA_Demands.csv', delimiter=";", decimal =',')
demands_2019 = pd.read_csv('dataset/2018_SCADA_Demands.csv', delimiter=";", decimal =',')

# Create a water network model
inp_file = 'dataset/L-TOWN.inp'
wn = wntr.network.WaterNetworkModel(inp_file)

dict_amr = dict({'AMR':{'Residential': ["n2", 'n44', 'n368', 'n3', 'n30', 'n356', 'n6', 'n17', 'n7', 'n364',
                                        'n8', 'n383', 'n9', 'n361', 'n10', 'n381', 'n11', 'n21', 'n13', 'n16', 'n371',
                                        'n18', 'n354', 'n19', 'n39', 'n20', 'n45', 'n373', 'n379', 'n22', 'n24', 'n32', 'n27', 'n358',
                                        'n29', 'n31', 'n36', 'n34', 'n355', 'n40', 'n360', 'n41', 'n42', 'n344', 'n349', 'n43', 'n345',
                                        'n346', 'n385', 'n350', 'n351', 'n386', 'n352', 'n353', 'n357', 'n362', 'n365', 'n375', 'n366',
                                        'n378','n366', 'n378', 'n369', 'n387', 'n372', 'n374', 'n376', 'n382', 'n384', 'n389'],
                        'Commercial': ['n26', 'n28', 'n35', 'n377']}})

dict_base_demand = {}
for j in wn.junctions():
    dict_base_demand[j[0]] = j[1].demand_timeseries_list.base_demand_list()

train_residential_df = pd.DataFrame(columns=['ds'])
train_residential_df['ds'] = demands_2018['Timestamp']

train_commercial_df = pd.DataFrame(columns=['ds'])
train_commercial_df['ds'] = demands_2018['Timestamp']

for n in dict_amr['AMR']['Residential']:
    base_demand = dict_base_demand[n][0]
    train_residential_df[n] = ((demands_2018[
                                    n] / 1000) / 3600) / base_demand  # L/s -> CMH documentation not correct: L/s instead of L/h
train_residential_df['y'] = train_residential_df.drop('ds', axis=1).mean(axis=1)
train_residential_df = train_residential_df[['ds', 'y']]

for n in dict_amr['AMR']['Commercial']:
    base_demand = dict_base_demand[n][1]
    train_commercial_df[n] = ((demands_2018[n] / 1000) / 3600) / base_demand  # L/h -> CMH
train_commercial_df['y'] = train_commercial_df.drop('ds', axis=1).mean(axis=1)
train_commercial_df = train_commercial_df[['ds', 'y']]

m_residential = NeuralProphet(yearly_seasonality=True)
m_residential.fit(train_residential_df)

future_residential = m_residential.make_future_dataframe(df=train_residential_df, n_historic_predictions = True)
forecast_residential = m_residential.predict(future_residential)

m_commercial = NeuralProphet(yearly_seasonality=True)
m_commercial.fit(train_commercial_df)

future_commercial = m_commercial.make_future_dataframe(df=train_commercial_df, n_historic_predictions = True)
forecast_commercial = m_commercial.predict(future_commercial)

predicted_demands= pd.DataFrame(columns=['ds']) #model predicts CMH
predicted_demands['ds'] = demands_2018['Timestamp']

for n in wn.junctions():
    if n[0] in dict_amr['AMR']['Residential'] or n[0] in dict_amr['AMR']['Commercial']:
        predicted_demands[n[0]] = (demands_2018[n[0]]/3600)/1000 #L/s ->CMH
    else:
        base_demand = dict_base_demand[n[0]]
        predicted_demands[n[0]] = forecast_residential[['yhat1']]*base_demand[0]
        if base_demand[1] > 0:
            predicted_demands[n[0]] = forecast_commercial[['yhat1']]*base_demand[1]
        if base_demand[2] > 0:
            predicted_demands[n[0]] = (demands_2018[n[0]]/3600)/1000 #L/s ->CMH #*base_demand[2]


# 2019
train_residential_df = pd.DataFrame(columns=['ds'])
train_residential_df['ds'] = demands_2019['Timestamp']

train_commercial_df = pd.DataFrame(columns=['ds'])
train_commercial_df['ds'] = demands_2019['Timestamp']

for n in dict_amr['AMR']['Residential']:
    base_demand = dict_base_demand[n][0]
    train_residential_df[n] = ((demands_2019[
                                    n] / 1000) / 3600) / base_demand  # L/s -> CMH documentation not correct: L/s instead of L/h
train_residential_df['y'] = train_residential_df.drop('ds', axis=1).mean(axis=1)
train_residential_df = train_residential_df[['ds', 'y']]

for n in dict_amr['AMR']['Commercial']:
    base_demand = dict_base_demand[n][1]
    train_commercial_df[n] = ((demands_2019[n] / 1000) / 3600) / base_demand  # L/h -> CMH
train_commercial_df['y'] = train_commercial_df.drop('ds', axis=1).mean(axis=1)
train_commercial_df = train_commercial_df[['ds', 'y']]

m_residential = NeuralProphet(yearly_seasonality=True)
m_residential.fit(train_residential_df)

future_residential = m_residential.make_future_dataframe(df=train_residential_df, n_historic_predictions = True)
forecast_residential = m_residential.predict(future_residential)

m_commercial = NeuralProphet(yearly_seasonality=True)
m_commercial.fit(train_commercial_df)

future_commercial = m_commercial.make_future_dataframe(df=train_commercial_df, n_historic_predictions = True)
forecast_commercial = m_commercial.predict(future_commercial)

predicted_demands_2019= pd.DataFrame(columns=['ds']) #model predicts CMH
predicted_demands_2019['ds'] = demands_2019['Timestamp']

for n in wn.junctions():
    if n[0] in dict_amr['AMR']['Residential'] or n[0] in dict_amr['AMR']['Commercial']:
        predicted_demands_2019[n[0]] = (demands_2019[n[0]]/3600)/1000 #L/s ->CMH
    else:
        base_demand = dict_base_demand[n[0]]
        predicted_demands_2019[n[0]] = forecast_residential[['yhat1']]*base_demand[0]
        if base_demand[1] > 0:
            predicted_demands_2019[n[0]] = forecast_commercial[['yhat1']]*base_demand[1]
        if base_demand[2] > 0:
            predicted_demands_2019[n[0]] = (demands_2019[n[0]]/3600)/1000 #L/s ->CMH #*base_demand[2]

print(predicted_demands_2019)

filename = 'dataset/predicted_demands_2018'
compression_options = dict(method='zip', archive_name=f'{filename}.csv')
predicted_demands.to_csv(f'{filename}.zip', compression=compression_options)

filename = 'dataset/predicted_demands_2019'
compression_options = dict(method='zip', archive_name=f'{filename}.csv')
predicted_demands_2019.to_csv(f'{filename}.zip', compression=compression_options)