import torch
import torchvision.transforms as transforms
from PIL import Image
from timm import create_model
import requests
import json

# Load Pretrained EfficientNet-B7
import torchvision.models as models

model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
model.eval()
#model = create_model('efficientnet_b7', pretrained=False)
#model.eval()  # Set to evaluation mode

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # EfficientNet-B7 expects ~600px images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and Process an Image
image_path = "Software/object_detection/Pictures/gurke.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Perform Inference
with torch.no_grad():
    output = model(image)

# Get Predicted Class
predicted_class = torch.argmax(output, dim=1).item()
print(f"Predicted Class ID: {predicted_class}")

LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels = requests.get(LABELS_URL).json()

# Klasse ausgeben
predicted_label = labels[str(predicted_class)][1]  # Holt den Klassennamen
print(f"Vorhergesagte Klasse: {predicted_label}")


