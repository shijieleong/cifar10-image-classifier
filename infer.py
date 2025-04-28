import torch
import torchvision.transforms as transforms
from PIL import Image

import utils
from neural_net import NeuralNet


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(), # Convert data values from 0~255 to 0~1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Convert data values from 0~1 to -1~1
    ])
    
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    return image
    


def main():
    # Define class name
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Load image
    image_paths = ['./images/sample1.jpg', './images/sample2.jpg', './images/sample3.jpg']
    images = [load_image(img) for img in image_paths]

    # Setup gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = NeuralNet()
    utils.load_checkpoint(net, './models/trained_model.pth')
    net.eval()

    with torch.no_grad():
        for image in images:
            image = image.to(device)
            output = net(image)
            _, predicted = torch.max(output, 1)
            
            print(f'Prediction: {class_names[predicted.item()]}')

    return


if __name__ == "__main__":
    main()



