from PIL import Image
import torch
from torchvision import transforms
import config
import numpy as np

img_transforms = transforms.Compose(
    [
        transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()
    ]
)

def load_model(model_definition, path_to_model):
    model = model_definition()
    model.load_state_dict(torch.load(path_to_model))
    model = model.to(config.DEVICE)
    return model

def predict_image(model, image_path):
    with torch.no_grad():
        image = Image.open(image_path)
        #Но нам нужно добавить еще одно измерение для тензора, чтобы он был (1, 3, 256, 256)
        image = img_transforms(image).unsqueeze(0)
        image = image.to(config.DEVICE)
        predMask = model(image)
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        predMask = (predMask > config.THRESHOLD) * 255
        #predMask = predMask * 255
        predMask = predMask.squeeze().astype(np.uint8)
        return predMask