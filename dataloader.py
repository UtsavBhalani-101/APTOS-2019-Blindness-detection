import torch
from PIL import Image
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        
        img_arr = np.array(pil_image)
        img_tensor = torch.Tensor(img_arr)
        
        return img_tensor, label        
    
    
dataset = APTOSData(inputs=images_path, target="train.csv", transforms=transformed)

print(dataset[0])

loader = DataLoader(dataset, batch_size=32, shuffle=True)

