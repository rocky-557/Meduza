# Meduza
# **Project: Diabetic Retinopathy Classification using BYOL and EfficientNet-B3**

## **Team Work Distribution**

### **Person 1: Data Preparation & Preprocessing**
#### **Step 1: Setup Environment**
1. Open Google Colab and create a new notebook.
2. Install dependencies:
   ```bash
   !pip install timm onnx onnxruntime torch torchvision albumentations
   ```

#### **Step 2: Download APTOS 2019 Dataset**
1. Obtain the Kaggle API key and upload `kaggle.json` to Colab.
2. Run:
   ```python
   import os
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle competitions download -c aptos2019-blindness-detection
   !unzip aptos2019-blindness-detection.zip -d data/
   ```

#### **Step 3: Data Preprocessing**
1. Load image paths and labels from `train.csv`.
2. Apply augmentations using Albumentations:
   ```python
   import pandas as pd
   import cv2
   import albumentations as A
   from albumentations.pytorch import ToTensorV2
   from torch.utils.data import Dataset, DataLoader
   
   df = pd.read_csv("data/train.csv")
   df['image_path'] = "data/train_images/" + df['id_code'] + ".png"
   
   transform = A.Compose([
       A.Resize(300, 300),
       A.HorizontalFlip(p=0.5),
       A.VerticalFlip(p=0.5),
       A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
       ToTensorV2(),
   ])
   ```

3. Create a dataset class:
   ```python
   class APTOSDataset(Dataset):
       def __init__(self, df, transform=None):
           self.df = df
           self.transform = transform
       
       def __len__(self):
           return len(self.df)
       
       def __getitem__(self, idx):
           image = cv2.imread(self.df.iloc[idx]['image_path'])
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           if self.transform:
               image = self.transform(image=image)['image']
           return image, image.clone()
   ```
4. Initialize DataLoader:
   ```python
   dataset = APTOSDataset(df, transform=transform)
   dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
   ```

---

### **Person 2: BYOL Model Implementation**
#### **Step 4: Define BYOL Model**
1. Load EfficientNet-B3 as a feature extractor:
   ```python
   import torch
   import torch.nn as nn
   import timm
   
   base_model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
   ```

2. Implement the BYOL projection head:
   ```python
   class MLP(nn.Module):
       def __init__(self, in_dim, out_dim=256):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(in_dim, 4096),
               nn.BatchNorm1d(4096),
               nn.ReLU(),
               nn.Linear(4096, out_dim)
           )
       def forward(self, x):
           return self.net(x)
   ```

3. Define BYOL model:
   ```python
   class BYOL(nn.Module):
       def __init__(self, base_model, feature_dim=256):
           super().__init__()
           self.online_encoder = nn.Sequential(base_model, MLP(1536, feature_dim))
           self.target_encoder = nn.Sequential(base_model, MLP(1536, feature_dim))
           for param in self.target_encoder.parameters():
               param.requires_grad = False
       
       def forward(self, x1, x2):
           z1_online = self.online_encoder(x1)
           z2_online = self.online_encoder(x2)
           z1_target = self.target_encoder(x1).detach()
           z2_target = self.target_encoder(x2).detach()
           return self.loss_fn(z1_online, z2_target) + self.loss_fn(z2_online, z1_target)
       
       def loss_fn(self, x, y):
           x = nn.functional.normalize(x, dim=-1)
           y = nn.functional.normalize(y, dim=-1)
           return 2 - 2 * (x * y).sum(dim=-1)
   ```

---

### **Person 3: Training & Fine-tuning**
#### **Step 5: Train BYOL Model**
```python
byol_model = BYOL(base_model).to("cuda")
optimizer = torch.optim.Adam(byol_model.parameters(), lr=3e-4)

for epoch in range(20):
    total_loss = 0
    for img1, img2 in dataloader:
        img1, img2 = img1.to("cuda"), img2.to("cuda")
        optimizer.zero_grad()
        loss = byol_model(img1, img2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {total_loss / len(dataloader):.4f}")
```

#### **Step 6: Fine-tune for Classification**
```python
class DRClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
```

---

### **Person 4: ONNX Conversion & Deployment**
#### **Step 7: Convert to ONNX**
```python
import torch.onnx

dummy_input = torch.randn(1, 3, 300, 300).to("cuda")
torch.onnx.export(DRClassifier(base_model), dummy_input, "model.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("Model saved to model.onnx")
```

#### **Step 8: Deploy on Mobile**
1. Use **ONNX Runtime** for inference.
2. Optimize model with quantization.
3. Integrate with Flutter or Android/iOS.

---
