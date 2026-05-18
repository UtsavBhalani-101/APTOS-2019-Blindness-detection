from torch.nn.modules import instancenorm
from torchvision import transforms
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import timm 

inp_path = "train_images/"

transformed = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])    
])


class customDataset(Dataset):
    def __init__(self, input_path, target_df, transformation, num_samples=None):
        self.inp = input_path
        self.df = target_df
        self.transformer = transformation
        if num_samples:
            self.df = self.df.head(num_samples)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id_code"]
        label = row["diagnosis"]
        
        img_path = os.path.join(self.inp, (image_id + '.png'))
        image = Image.open(img_path).convert("RGB")
        
        image = self.transformer(image)
           
        return image, label
    

df = pd.read_csv("train.csv")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])

print(train_df.shape)
print(val_df.shape)

train_dataset = customDataset(inp_path, train_df, transformed, 20)
val_dataset = customDataset(inp_path, val_df, transformed, 20)

train_loader = DataLoader(train_dataset, shuffle=True)
val_loader = DataLoader(val_dataset, shuffle=True)


def validation_loop(loader, model):
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for img, label in loader:
        
        output = model(img)
        loss = criterion(output, label)
        pred = torch.argmax(output, dim=1)
        
        running_loss += loss.item()
        all_preds.append(pred)
        all_labels.append(loss)
        
    val_loss = running_loss / len(loader)



            
        
    

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()   # clear old gradients first
        
        # 1. Forward pass
        predictions = model(X_batch)
        
        # 2. Compute loss
        loss = criterion(predictions, y_batch)
        
        # 3. Backward pass
        loss.backward()         # compute new gradients
        
        # 4. Update weights
        optimizer.step()
    
        running_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    
