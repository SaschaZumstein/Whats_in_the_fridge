import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')



efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

efficientnet.eval().to(device)

uris = [
    'https://gemuesebodensee.ch/sites/default/files/styles/spalte_rechts_1_3/public/produktebilder/Gurke_kl_0.jpg?itok=GIDvNZzG',
    'https://chefsmandala.com/wp-content/uploads/2018/03/Squash-Zucchini.jpg',
    'https://i.pinimg.com/736x/55/f0/6a/55f06adb0295765e2e120249e817f78f.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

# Convert all images to RGB before creating the batch
batch = torch.cat(
    [utils.prepare_input_from_uri(uri.convert('RGB') if isinstance(uri, Image.Image) else uri) for uri in uris]
).to(device)


with torch.no_grad():
    output = torch.nn.functional.softmax(efficientnet(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)

for uri, result in zip(uris, results):
    # Open the image directly or from the URL
    img = Image.open(requests.get(uri, stream=True).raw) if isinstance(uri, str) else uri
    img.thumbnail((256,256), Image.LANCZOS)
    plt.imshow(img)
    plt.show()
    print(result)