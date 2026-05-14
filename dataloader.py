import torch
from PIL import Image
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import timm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optional: Print the name of the GPU being used
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

images_path = r"train_images/"
train_path = r"train.csv"

df = pd.read_csv(train_path)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)


transformed = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


class APTOSData(Dataset):

    def __init__(self, inputs, target, transforms, num_samples=None):
        self.img_path = inputs
        self.df = target.reset_index(drop=True)  # reset the split and index 
        if num_samples is not None:
            self.df = self.df.head(num_samples)
        self.transform = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        label = row['diagnosis']
        
        img_path = os.path.join(self.img_path, img_id + '.png')   # cache path if possible
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label       
    
    
train_dataset = APTOSData(inputs=images_path, target=train_df, transforms=transformed)
val_dataset = APTOSData(inputs=images_path, target=val_df, transforms=transformed)


print(train_dataset[0])

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4,           # ← Change this
    pin_memory=True,         # ← Add this (very important for GPU)
    persistent_workers=True,  # Optional, helps on Kaggle
    prefetch_factor=2          # ← Also good to try
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False,           # shuffle wastes compute and doesn't matter for testing
    num_workers=4,           # ← Change this
    pin_memory=True,         # ← Add this (very important for GPU)
    persistent_workers=True,  # Optional, helps on Kaggle
    prefetch_factor=2          # ← Also good to try
)

def evaluate(model, loader, device):
    model.eval()  # switch off dropout, batchnorm uses running stats
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():  # no gradient computation, saves memory
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    print(f"Val Loss: {epoch_loss:.4f} | Val QWK: {qwk:.4f}")
    print(confusion_matrix(all_labels, all_preds))
    
    return qwk

# & setting up weights

# labels = [0]*1805 + [1]*370 + [2]*999 + [3]*193 + [4]*295

labels= train_df['diagnosis'].values
weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2,3,4]), y=labels)

class_weights = torch.FloatTensor(weights).to(device)

print(class_weights)

# & setting up loss function

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C] (logits), targets: [N] (indices)
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# & training pipeline requirements


model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = FocalLoss(weight=class_weights, gamma=1.0)

# & training pipeline loop


print("training loop start...")

# for epoch in range(5):

#     model.train()

#     all_preds = []
#     all_labels = []

#     running_loss = 0.0

#     for batch_idx, (images, labels) in enumerate(train_loader):

#         images = images.to(device)
#         labels = labels.to(device)

#         print(f"Epoch {epoch} - Starting batch {batch_idx + 1}/{len(train_loader)}...")

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch {epoch} - Batch {batch_idx + 1}/{len(train_loader)} done. Loss: {loss.item():.4f}")

#         running_loss += loss.item()

#     train_loss = running_loss / len(train_loader)
    
#     # --- validation ---
#     val_qwk = evaluate(model, val_loader, device)
    
#     print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val QWK: {val_qwk:.4f}")

#     print(f"Epoch {epoch} done")

