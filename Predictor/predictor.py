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
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ml_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = 'E:\source\ISS\AD_project\ML\data\hdbdata.csv'
encoder_dir = os.path.join(ml_directory, 'encoder')

def train_model(model, criterion, optimizer, device, train_loader, val_loader ,num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()  
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  
            outputs = model(inputs)  
            predicted_labels = outputs[6][:, :, 4]
            loss = criterion(predicted_labels, labels)  
            loss.backward()  
            optimizer.step()
            

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions[6][:, :, 4], y_batch)
                total_val_loss += loss.item()
    
        average_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {average_val_loss}")
        
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
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

def create_model():
    model = iTransformer(
    num_variates = 5,
    lookback_len = 12,                  # or the lookback length in the paper
    dim = 256,                          # model dimensions
    depth = 6,                          # depth
    heads = 8,                          # attention heads
    dim_head = 64,                      # head dimension
    pred_length = (6),     # can be one prediction, or many
    num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
    use_reversible_instance_norm = True # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
    )
    return model

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
    df[['floor_area_sqm']] = scaler.fit_transform(df[['floor_area_sqm']])

    sta_scalar = StandardScaler()
    df[['resale_price']] = sta_scalar.fit_transform(df[['resale_price']])

    #time series
    df['year'] = pd.to_datetime(df['month']).dt.year
    df['month_num'] = pd.to_datetime(df['month']).dt.month
    df['time_index'] = df['year'] * 12 + df['month_num'] - df['year'].min() * 12
    
    split_time_index = df['time_index'].quantile(0.8, interpolation='nearest')

    train_df = df[df['time_index'] <= split_time_index]
    test_df = df[df['time_index'] > split_time_index]

    
    X_train, y_train = build_sequences(train_df, time_series_length=12, forecast_length=6)
    X_test, y_test = build_sequences(test_df, time_series_length=12, forecast_length=6)
    
    X_train_tensor = torch.tensor(list(X_train), dtype=torch.float)
    y_train_tensor = torch.tensor(list(y_train), dtype=torch.float)
    X_test_tensor = torch.tensor(list(X_test), dtype=torch.float)
    y_test_tensor = torch.tensor(list(y_test), dtype=torch.float)

    joblib.dump(town_encoder, os.path.join(encoder_dir, 'town_encoder.joblib'))
    joblib.dump(flat_type_encoder, os.path.join(encoder_dir, 'flat_type_encoder.joblib'))
    joblib.dump(scaler, os.path.join(encoder_dir, 'minmax_scaler.joblib'))
    joblib.dump(sta_scalar, os.path.join(encoder_dir, 'standard_scaler.joblib'))

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def build_sequences(df, time_series_length=12, forecast_length=6):
    X, y = [], []
    
    for town, group in df.groupby('town'):
        group = group.sort_values(by='time_index')

        for start_idx in range(len(group) - time_series_length - forecast_length + 1):
            X_seq = group.iloc[start_idx:start_idx+time_series_length][['flat_type', 'floor_area_sqm','town','year','resale_price']].values
            y_seq = group.iloc[start_idx+time_series_length:start_idx+time_series_length+forecast_length]['resale_price'].values
            
            X.append(X_seq)
            y.append(y_seq)
        
    X = np.array(X, dtype=object) 
    y = np.array(y, dtype=object)

    return X, y

# def padding(source_df, x):
#     town_avg_prices = source_df.groupby('town')['resale_price'].mean()
#     x['resale_price'] = x.apply(
#     lambda row: town_avg_prices[row['town']] if pd.isnull(row['resale_price']) else row['resale_price'],
#     axis=1
#     )

def fill_time_series_with_past_average( input_data, time_series_length=12, area_tolerance=5):
    df = pd.read_csv(data_path)

    df = df.drop(columns=['block','street_name','storey_range','flat_model','lease_commence_date','remaining_lease'])
    df['year'] = pd.to_datetime(df['month']).dt.year
    df['month_num'] = pd.to_datetime(df['month']).dt.month
    df['time_index'] = df['year'] * 12 + df['month_num'] - df['year'].min() * 12
    
    time_series = []
    
    input_town = input_data['town']
    input_floor_area = input_data['floor_area_sqm']
    
    df_sorted = df.sort_values(by='time_index', ascending=False)
    
    for time_point in df_sorted['time_index'].unique()[:time_series_length]:
        similar_records = df_sorted[
            (df_sorted['time_index'] == time_point) &
            (df_sorted['town'] == input_town) &
            (df_sorted['floor_area_sqm'] >= input_floor_area - area_tolerance) &
            (df_sorted['floor_area_sqm'] <= input_floor_area + area_tolerance)
        ]
        
        avg_price = similar_records['resale_price'].mean()
        
        if pd.isna(avg_price):
            avg_price = input_data['resale_price']
        
        time_series.append(avg_price)
    
    while len(time_series) < time_series_length:
        time_series.append(time_series[-1])
    
    time_series = time_series[:time_series_length]
    
    time_series.reverse()
    
    return time_series

def predict(model_path, input_data):
    sta_scalar = joblib.load(os.path.join(encoder_dir, 'standard_scaler.joblib'))
    
    model = iTransformer(
    num_variates = 5,
    lookback_len = 12,                  # or the lookback length in the paper
    dim = 256,                          # model dimensions
    depth = 6,                          # depth
    heads = 8,                          # attention heads
    dim_head = 64,                      # head dimension
    pred_length = (6),     # can be one prediction, or many
    num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
    use_reversible_instance_norm = True # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
    )

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    input_tensor = torch.tensor(input_data, dtype=torch.float).to(device)
    
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_tensor)

    predictions_np = predictions[6][:, :, 4].cpu().numpy()

    predictions_unscaled = sta_scalar.inverse_transform(predictions_np)

    return predictions_unscaled

def prepare_data(input_data):
    town_encoder = joblib.load(os.path.join(encoder_dir, 'town_encoder.joblib'))
    flat_type_encoder = joblib.load(os.path.join(encoder_dir, 'flat_type_encoder.joblib'))
    scaler = joblib.load(os.path.join(encoder_dir, 'minmax_scaler.joblib'))
    sta_scalar = joblib.load(os.path.join(encoder_dir, 'standard_scaler.joblib'))

    series = fill_time_series_with_past_average(input_data)

    town_encoded = town_encoder.transform([input_data['town']])[0]
    flat_type_encoded = flat_type_encoder.transform([input_data['flat_type']])[0]
    floor_area_sqm_scaled = scaler.transform([[input_data['floor_area_sqm']]])[0][0]
    year = input_data['year']
    extended_features_time_series = []
    for price in series:
        price_scaled = sta_scalar.transform([[price]])[0][0]
        features = [flat_type_encoded, floor_area_sqm_scaled, town_encoded, year, price_scaled]
        extended_features_time_series.append(features)

    extended_features_time_series_np = np.array(extended_features_time_series)

    return extended_features_time_series_np

def predict_api(input_data):    
    model_path = 'best_model.pth'
    input_series = prepare_data(input_data)
    prediction = predict(model_path, input_series)
    
    return prediction

if __name__ == '__main__':
    # train
    # X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = process_data(data_path)

    # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model = create_model()

    # model.to(device)

    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # model = train_model(model, criterion, optimizer, device, train_loader, val_loader, num_epochs=10)

    # for test
    
    input_data = {
        'town': 'ANG MO KIO',
        'floor_area_sqm': 67,
        'flat_type': '3 ROOM',
        'year': 2023,
        'resale_price': 440000,
    }
    prediction = predict_api(input_data)
    print(prediction)