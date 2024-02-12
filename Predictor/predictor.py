import iTransformer.iTransformer as iTransformer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train_model(model, criterion, optimizer, device, train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  
            predictions = model(inputs) 
            loss = criterion(predictions.squeeze(), labels)  
            loss.backward()
            optimizer.step()  
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    return model

def eval_model(model, test_loader, criterion, device):
    model.eval()  
    test_loss = 0

    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = criterion(predictions.squeeze(), labels)  
            test_loss += loss.item()
        
    avg_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.4f}')
    return avg_loss

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def process_data(data_path):
    df = pd.read_csv(data_path)
    
    #drop
    df = df.drop(columns=['block','street_name','storey_range','flat_model','lease_commence_date','remaining_lease'])
    
    #encode
    town_encoder = LabelEncoder()
    df['town'] = town_encoder.fit_transform(df['town'])

    flat_type_encoder = LabelEncoder()
    df['flat_type'] = flat_type_encoder.fit_transform(df['flat_type'])

    #scale
    scaler = MinMaxScaler()
    df[['floor_area_sqm', 'resale_price']] = scaler.fit_transform(df[['floor_area_sqm', 'resale_price']])

    #time series
    df['year'] = pd.to_datetime(df['month']).dt.year
    df['month_num'] = pd.to_datetime(df['month']).dt.month
    df['time_index'] = df['year'] * 12 + df['month_num'] - df['year'].min() * 12
    
    
    X, y = build_sequences(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def build_sequences(df, time_series_length=12, forecast_length=3):
    X, y = [], []

    features = ['town', 'flat_type', 'floor_area_sqm', 'resale_price', 'year', 'month_num']
    
    for i in range(df['time_index'].min(), df['time_index'].max() - time_series_length - forecast_length + 1):
        mask = (df['time_index'] >= i) & (df['time_index'] < i + time_series_length)
        forecast_mask = (df['time_index'] >= i + time_series_length) & (df['time_index'] < i + time_series_length + forecast_length)
        
        if mask.sum() == time_series_length and forecast_mask.sum() == forecast_length:

            X.append(df.loc[mask, features].values)
            
            y.append(df.loc[forecast_mask, 'resale_price'].values)

    X = np.array(X)
    y = np.array(y)

    return X, y
