import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pretrained ResNet-18
model = models.resnet18(pretrained=True)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img_path = "../images/test/houndrack3.jpg"
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)

# Load class labels
with open("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Predicted class: {labels[predicted.item()]}")
