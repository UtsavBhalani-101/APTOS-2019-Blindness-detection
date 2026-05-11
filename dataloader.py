import torch
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import timm

images_path = r"train_images/"


transformed = transforms.Compose([
    transforms.Resize((224, 224)),
    
    transforms.ToTensor()
])




class APTOSData(Dataset):

    def __init__(self, inputs, target, transforms):
        self.img_path = inputs
        
        self.df = pd.read_csv(target)
        
        self.transform = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # given index, get the image tensor and it's target
        row = self.df.iloc[idx]
        
        img_id = row["id_code"]
        label = row["diagnosis"]
        
        img_id = img_id + ".png"
        image = os.path.join(self.img_path, img_id)
        pil_image = Image.open(image)
        
        transformed_img = self.transform(pil_image)
        
        return transformed_img, label        
    
    
dataset = APTOSData(inputs=images_path, target="train.csv", transforms=transformed)

print(dataset[0])

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# & training pipeline requirements

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=5)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# & training pipeline loop


print("training loop start...")
for epoch in range(1):
    model.train()  # tells model it's in training mode
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Epoch {epoch} - Starting batch {batch_idx + 1}/{len(train_loader)}...")
        
        optimizer.zero_grad()      # clear gradients from last step
        
        outputs = model(images)    # forward pass: (32, 5) logits
        loss = criterion(outputs, labels)  # scalar loss
        
        loss.backward()            # compute gradients
        optimizer.step()           # update weights
        
        print(f"Epoch {epoch} - Batch {batch_idx + 1}/{len(train_loader)} done. Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch} done")