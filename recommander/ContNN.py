import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

current_script_path = os.path.abspath(__file__)

ml_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

data_directory = os.path.join(ml_directory, 'data')

data_path = os.path.join(data_directory, 'property_data.csv')
data_path ="E:\source\ISS\AD_project\ML\data\property_data.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def props_data_process():
    df = pd.read_csv(data_path)

    # X = df.drop(columns=['user_id', 'property_id', 'interest_level'])


    town_encoder = LabelEncoder()
    df['preferred_town_encoded'] = town_encoder.fit_transform(df['preferred_town'])
    df['town_encoded'] = town_encoder.transform(df['town']) 

    flat_type_encoder = LabelEncoder()
    df['preferred_flat_type_encoded'] = flat_type_encoder.fit_transform(df['preferred_flat_type'])
    df['flat_type_encoded'] = flat_type_encoder.transform(df['flat_type'])

    flat_model_encoder = LabelEncoder()
    df['flat_model_encoded'] = flat_model_encoder.fit_transform(df['flat_model'])

    bool_columns = ['town_match', '3-room_match', '4-room_match', 'Executive_match', '2-room_match', 'price_in_range']
    df[bool_columns] = df[bool_columns].astype(int)
    
    scaler = MinMaxScaler()
    numeric_columns = ['low_price', 'high_price', 'floor_area_sqm', 'resale_price']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    df = df.drop(columns=['preferred_town', 'preferred_flat_type', 'town', 'flat_type', 'flat_model',])
    y = df['interest_level']
    X = df.drop(columns=['user_id', 'property_id', 'interest_level'])
    # X.to_csv('X.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)  # 调整形状以匹配模型的期望输入
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


    
def train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = DNN(X_train_tensor.shape[1])
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20 
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
                    predicted = outputs.squeeze() >= 0.5 
                    correct += (predicted == labels.view(-1)).type(torch.float).sum().item()
            
            test_loss /= len(test_loader.dataset)
            accuracy = correct / len(test_loader.dataset)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    torch.save(model.state_dict(), 'DNN_model_parameters.pth')

def predict(model, X):
    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(X)
    return outputs

def load_model(model_path, input_size):
    model = DNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

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

