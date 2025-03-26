import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1️⃣ Load Pretrained DeepLabV3+ Model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)  # ResNet-101 backbone
model.eval()  # Set to evaluation mode

# 2️⃣ Load & Preprocess the Image
image_path = "Software/object_detection/Pictures/fridge_0001.jpeg"  # Replace with your image
image = Image.open(image_path).convert("RGB")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Apply transformation and add batch dimension
input_image = transform(image).unsqueeze(0)

# 3️⃣ Run Segmentation
with torch.no_grad():
    output = model(input_image)["out"]  # Get model output

# Get the predicted segmentation mask
segmentation_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# 4️⃣ Overlay Segmentation on Original Image
def overlay_segmentation(image, mask, alpha=0.5):
    """Overlays the segmentation mask on the original image."""
    cmap = plt.get_cmap("jet")  # Colormap for visualization
    mask_colored = cmap(mask / np.max(mask))[:, :, :3]  # Normalize & apply colormap
    blended = alpha * np.array(image) / 255 + (1 - alpha) * mask_colored * 255
    return blended.astype(np.uint8)

# Apply overlay
segmented_image = overlay_segmentation(image, segmentation_map)

# Show results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(segmented_image)
ax[1].set_title("Segmented Image")
ax[1].axis("off")

plt.show()
