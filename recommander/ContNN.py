import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

current_script_path = os.path.abspath(__file__)

ml_directory = os.path.dirname(os.path.dirname(current_script_path))

data_directory = os.path.join(ml_directory, 'data')

data_path = os.path.join(data_directory, 'property_data.csv')
data_path ="E:\source\ISS\AD_project\ML\data\property_data.csv"
encoder_dir = os.path.join(ml_directory, 'encoder')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def props_data_process():
    df = pd.read_csv(data_path)

    # X = df.drop(columns=['user_id', 'property_id', 'interest_level'])
    
    town_encoder = LabelEncoder()
    df['preferred_town_encoded'] = town_encoder.fit_transform(df['preferred_town'])
    df['town_encoded'] = town_encoder.transform(df['town']) 

    flat_type_encoder = LabelEncoder()
    df['flat_type_encoded'] = flat_type_encoder.fit_transform(df['flat_type'])

    flat_model_encoder = LabelEncoder()
    df['flat_model_encoded'] = flat_model_encoder.fit_transform(df['flat_model'])

    bool_columns = ['town_match', '3 ROOM_match', '4 ROOM_match', 'EXECUTIVE_match', '2 ROOM_match','1 ROOM_match','5 ROOM_match',"MULTI-GENERATION_match", 'price_in_range']
    df[bool_columns] = df[bool_columns].astype(int)
    
    scaler = MinMaxScaler()
    numeric_columns = ['low_price', 'high_price', 'floor_area_sqm', 'resale_price']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    df = df.drop(columns=['preferred_town', 'preferred_flat_type', 'town', 'flat_type', 'flat_model',])
    y = df['interest_level']
    X = df.drop(columns=['user_id', 'property_id', 'interest_level'])
    X.to_csv('X.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)  # 调整形状以匹配模型的期望输入
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

    joblib.dump(town_encoder, 'town_encoder.pkl')
    joblib.dump(flat_type_encoder, 'flat_type_encoder.pkl')
    joblib.dump(flat_model_encoder, 'flat_model_encoder.pkl')
    joblib.dump(scaler, 'minmax_scaler.pkl')

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def prepare_data(input_json):
    town_encoder = joblib.load(os.path.join(encoder_dir, 'town_encoder.pkl'))
    flat_type_encoder = joblib.load(os.path.join(encoder_dir, 'flat_type_encoder.pkl'))
    flat_model_encoder = joblib.load(os.path.join(encoder_dir, 'flat_model_encoder.pkl'))
    scaler = joblib.load(os.path.join(encoder_dir, 'minmax_scaler.pkl'))
    
    items_features_df = pd.read_csv(os.path.join(data_directory, 'property.csv'))
    items_features_df['town_encoded'] = town_encoder.transform(items_features_df['town'])
    items_features_df['flat_type_encoded'] = flat_type_encoder.transform(items_features_df['flat_type'])
    items_features_df['flat_model_encoded'] = flat_model_encoder.transform(items_features_df['flat_model'])

    flat_types = items_features_df['flat_type'].unique().tolist()

    df = pd.DataFrame([input_json])
    df['preferred_town_encoded'] = town_encoder.transform([input_json['preferred_town']])
    preferred_flat_types = input_json['preferred_flat_type']
    for flat_type in flat_types:
        df[f'prefers_{flat_type}'] = df.apply(lambda x: 1 if flat_type in preferred_flat_types else 0, axis=1)

    df_expanded = pd.concat([df]*len(items_features_df), ignore_index=True)

    df_expanded['town_match'] = (df_expanded['preferred_town_encoded'] == items_features_df['town_encoded']).astype(int)
    df_expanded['price_in_range'] = ((df_expanded['low_price'] <= items_features_df['resale_price']) & (items_features_df['resale_price'] <= df_expanded['high_price'])).astype(int)


    combined_features_df = pd.concat([df_expanded.reset_index(drop=True), items_features_df.reset_index(drop=True)], axis=1)
    

    for flat_type in flat_types:
        combined_features_df[f'{flat_type}_match'] = combined_features_df['flat_type_encoded'].apply(lambda x: 1 if flat_type in preferred_flat_types else 0)

    numeric_columns = ['low_price', 'high_price', 'floor_area_sqm', 'resale_price']
    combined_features_df[numeric_columns] = scaler.transform(combined_features_df[numeric_columns])


    final_columns_order = ['low_price', 'high_price', 'prefers_2 ROOM', 'prefers_3 ROOM', 'prefers_4 ROOM', 
                           'prefers_5 ROOM', 'prefers_EXECUTIVE', 'prefers_1 ROOM', 'prefers_MULTI-GENERATION', 
                           'floor_area_sqm', 'resale_price', 'town_match', '2 ROOM_match', '3 ROOM_match', 
                           '4 ROOM_match', '5 ROOM_match', 'EXECUTIVE_match', '1 ROOM_match', 'MULTI-GENERATION_match', 
                           'price_in_range', 'preferred_town_encoded', 'town_encoded', 
                           'flat_type_encoded', 'flat_model_encoded']
    
    model_features = combined_features_df[final_columns_order]

    combined_features_tensor = torch.tensor(model_features.values, dtype=torch.float)

    return combined_features_tensor, combined_features_df['property_id'].values

def train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = DNN(X_train_tensor.shape[1])
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 4 
    for epoch in range(num_epochs):
            model.train()  
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels.view(-1))
                loss.backward() 
                optimizer.step()  
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)

            model.eval()  
            test_loss = 0.0
            correct = 0
            
            with torch.no_grad():  
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1), labels.view(-1))
                    test_loss += loss.item() * inputs.size(0)
            
            test_loss /= len(test_loader.dataset)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    torch.save(model.state_dict(), 'DNN_model_parameters.pth')

def predict(model, X):
    model.eval()
    model.to(device)
    X = X.to(device)
    with torch.no_grad():
        outputs = model(X)
    return outputs

def load_model(model_path, input_size):
    model = DNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def recommend_top_n_items(user_features, top_n = 10, device=device):
    user_features_tensor, property_ids = prepare_data(user_features)
    model = load_model('DNN_model_parameters.pth', user_features_tensor.shape[1])
    with torch.no_grad():
        predictions = model(user_features_tensor).squeeze()  # 假设输出是单维度的分数

    top_n = min(top_n, predictions.size(0))

    _, top_indices = torch.topk(predictions, top_n)

    top_property_ids = property_ids[top_indices.cpu().numpy()]
    print(top_property_ids)
    return top_property_ids

class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))  
        return x
    

if __name__ =="__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    #only for train

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = props_data_process()
    train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    user_features = {
        'user_id': 1,
        'preferred_town': 'BISHAN',
        'preferred_flat_type': {'4 ROOM', '5 ROOM'},
        'low_price': 300000,
        'high_price': 500000,
    }

    # user_features_tensor, property_ids = prepare_data(user_features)
    top = recommend_top_n_items(user_features)

