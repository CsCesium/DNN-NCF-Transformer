import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
current_script_path = os.path.abspath(__file__)
ml_directory = os.path.dirname(os.path.dirname(current_script_path))
encoder_dir = os.path.join(ml_directory, 'encoder')
data_path ="E:\source\ISS\AD_project\ML\data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExtendedNCF(nn.Module):
    def __init__(self, num_users, num_items, num_item_features, embedding_size, layers=[64, 32, 16, 8]):
        super(ExtendedNCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        self.item_feature_layers = nn.ModuleList()
        current_input_size = num_item_features
        for layer_size in layers[:-1]:  
            self.item_feature_layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
        
        # MLP
        self.MLP_layers = nn.ModuleList()
        mlp_input_size = embedding_size * 2 + layers[-2]  
        for in_size, out_size in zip([mlp_input_size] + layers[:-1], layers):
            self.MLP_layers.append(nn.Linear(in_size, out_size))
        
        self.output_layer = nn.Linear(layers[-1], 1)
        
    def forward(self, user_ids, item_ids, item_features):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        
        for layer in self.item_feature_layers:
            item_features = F.relu(layer(item_features))
        
        interaction = torch.cat((user_embedding, item_embedding, item_features), -1)
        
        for layer in self.MLP_layers:
            interaction = F.relu(layer(interaction))
        
        prediction = torch.sigmoid(self.output_layer(interaction))
        return prediction
    
def process_data():
    # Load data
    interactions_df = pd.read_csv(data_path + '\interaction.csv')
    property_df = pd.read_csv(data_path + '\property.csv')

    merged_df = pd.merge(interactions_df, property_df, on='property_id')

    # Preprocess data
    #id shall start from 0
    merged_df['user_id'] = merged_df['user_id'] - 1
    merged_df['property_id'] = merged_df['property_id'] - 1
    town_encoder = LabelEncoder()
    merged_df['town_encoded'] = town_encoder.fit_transform(merged_df['town'])

    flat_type_encoder = LabelEncoder()
    merged_df['flat_type_encoded'] = flat_type_encoder.fit_transform(merged_df['flat_type'])

    flat_type_encoder = LabelEncoder()
    merged_df['flat_type_encoded'] = flat_type_encoder.fit_transform(merged_df['flat_type'])

    scaler = MinMaxScaler()
    merged_df[['floor_area_sqm', 'resale_price','interaction_count']] = scaler.fit_transform(merged_df[['floor_area_sqm', 'resale_price','interaction_count']])
    
    X_user = torch.tensor(merged_df['user_id'].values)
    X_item = torch.tensor(merged_df['property_id'].values)
    X_item_features = torch.tensor(merged_df[['town_encoded', 'flat_type_encoded', 'floor_area_sqm', 'resale_price']].values, dtype=torch.float)
    y = torch.tensor(merged_df['interaction_count'].values, dtype=torch.float)

    X_user_np = X_user.numpy()
    X_item_np = X_item.numpy()
    X_item_features_np = X_item_features.numpy()
    y_np = y.numpy()


    X_combined_np = np.hstack((X_user_np.reshape(-1, 1), X_item_np.reshape(-1, 1), X_item_features_np))

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_combined_np, y_np, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train_np, dtype=torch.float)
    y_train = torch.tensor(y_train_np, dtype=torch.float)
    X_test = torch.tensor(X_test_np, dtype=torch.float)
    y_test = torch.tensor(y_test_np, dtype=torch.float)

    X_train_user = X_train[:, 0].long()  
    X_train_item = X_train[:, 1].long()  
    X_train_item_features = X_train[:, 2:]  

    X_test_user = X_test[:, 0].long()
    X_test_item = X_test[:, 1].long()
    X_test_item_features = X_test[:, 2:]

    train_dataset = TensorDataset(X_train_user, X_train_item, X_train_item_features, y_train)
    test_dataset = TensorDataset(X_test_user, X_test_item, X_test_item_features, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    joblib.dump(town_encoder, 'town_encoder_ncf.joblib')
    joblib.dump(flat_type_encoder, 'flat_type_encoder_ncf.joblib')
    joblib.dump(scaler, 'scaler_ncf.joblib')    

    return train_loader, test_loader

def train_model(train_loader, model, criterion, optimizer, device):
    num_epochs = 10 
    for epoch in range(num_epochs):
        model.train()  
        total_loss = 0
        
        for user_ids, item_ids, item_features, labels in train_loader:
            user_ids, item_ids, item_features, labels = user_ids.to(device), item_ids.to(device), item_features.to(device), labels.to(device)
            
            optimizer.zero_grad()  
            predictions = model(user_ids, item_ids, item_features) 
            loss = criterion(predictions.squeeze(), labels)  
            loss.backward()
            optimizer.step()  
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    save_model(model)

def save_model(model):
    torch.save(model.state_dict(), 'NCF_model.pth')

def eval_model(model, test_loader, criterion, device):
    model.eval()  
    test_loss = 0

    with torch.no_grad(): 
        for user_ids, item_ids, item_features, labels in test_loader:
            user_ids, item_ids, item_features, labels = user_ids.to(device), item_ids.to(device), item_features.to(device), labels.to(device)
            predictions = model(user_ids, item_ids, item_features)
            loss = criterion(predictions.squeeze(), labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')

def predict(model, user_id, property_id, town, flat_type, floor_area_sqm, resale_price, town_encoder, flat_type_encoder, scaler):
    town_encoded = town_encoder.transform([town])
    flat_type_encoded = flat_type_encoder.transform([flat_type])
    item_features = np.array([[floor_area_sqm, resale_price]])
    item_features_scaled = scaler.transform(item_features)
    
    user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
    property_id_tensor = torch.tensor([property_id], dtype=torch.long).to(device)
    item_features_tensor = torch.tensor(np.hstack((town_encoded, flat_type_encoded, item_features_scaled)), dtype=torch.float).to(device)
    
    model.eval()  
    with torch.no_grad():
        prediction = model(user_id_tensor, property_id_tensor, item_features_tensor)
    
    return prediction.cpu().numpy()

def recommend_top_n_items(model, user_id, items_df, n, device):
    #encode town and flat_type
    town_encoder = joblib.load(os.path.join(encoder_dir, 'town_encoder_ncf.joblib'))
    flat_type_encoder = joblib.load(os.path.join(encoder_dir, 'flat_type_encoder_ncf.joblib'))
    scaler = MinMaxScaler()

    user_id = user_id - 1
    items_df['property_id'] = items_df['property_id'] - 1
    scaler.fit(items_df[['floor_area_sqm', 'resale_price']])
    items_df['town_encoded'] = town_encoder.transform(items_df['town'])
    items_df['flat_type_encoded'] = flat_type_encoder.transform(items_df['flat_type'])
    items_df[['floor_area_sqm', 'resale_price']] = scaler.transform(items_df[['floor_area_sqm', 'resale_price']])


    user_id_tensor = torch.tensor([user_id] * len(items_df), dtype=torch.long).to(device)
    
    item_ids_tensor = torch.tensor(items_df['property_id'].values, dtype=torch.long).to(device)
    item_features_tensor = torch.tensor(items_df[['town_encoded', 'flat_type_encoded', 'floor_area_sqm', 'resale_price']].values, dtype=torch.float).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(user_id_tensor, item_ids_tensor, item_features_tensor).squeeze()
    
    item_scores = predictions.cpu().numpy()
    item_ids = items_df['property_id'].values
    
    top_n_item_indices = item_scores.argsort()[-n:][::-1]
    top_n_item_ids = item_ids[top_n_item_indices]
    top_n_item_ids = top_n_item_ids + 1
    return top_n_item_ids

def load_model(model):
    model.load_state_dict(torch.load('NCF_model.pth'))
    return model

def nfc_recommander(user_id):
    model = ExtendedNCF(num_users=1000, num_items=1500, num_item_features=4, embedding_size=20).to(device)
    model.to(device)
    
    items_df = pd.read_csv(data_path + '\property.csv')

    load_model(model)
    
    return recommend_top_n_items(model, user_id, items_df, 10, device)

    

if __name__ == "__main__":
    model = ExtendedNCF(num_users=1000, num_items=1500, num_item_features=4, embedding_size=20).to(device)
    criterion = torch.nn.BCELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader, test_loader = process_data()
    
    # train_model(train_loader, model, criterion, optimizer, device)
    
    # eval_model(model, test_loader, criterion, device)

    print(nfc_recommander(1))