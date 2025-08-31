from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

to_pil = transforms.ToPILImage()

# Function that take in input the image path, a trained network, and a transform function
# and uses the given net to predict the image class
def predict_img(img_path, net, transform):
    img = Image.open(img_path).convert("L")
    plt.imshow(img)
    img = transform(img)
    img = img.unsqueeze(0)

    net.eval()
    with torch.no_grad():
        outputs = net(img)
        _, predicted = torch.max(outputs.data, 1)
        print(f'the predicted class is: {predicted[0]}')