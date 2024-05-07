import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from trainer import Classifier

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model('./TumorModel.pth')

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),  # Convert grayscale images to RGB
        transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.ToTensor(),  # Convert the image to a tensor
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

def predict(image):
    with torch.no_grad():
        image = image.to(device)  # Ensure the image tensor is on the correct device
        output = model(image)  # Forward pass
        _, predicted = torch.max(output, 1)  # Get the predicted class
    return predicted.item()
