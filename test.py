import cv2 as cv
import torch
from torchvision import transforms
import torchvision.transforms as TF
from classification_model import Mask_CNN
import matplotlib.pyplot as plt
    
def test(state_dict_path, image):
    trans = TF.Compose([
    transforms.ToPILImage(),
    transforms.Resize((60, 60)),
    transforms.ToTensor()
    ])
    model = Mask_CNN()
    model.load_state_dict(torch.load(state_dict_path))
    img = cv.imread(image)
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        y = model(img)
        output = torch.argmax(y, dim=1)
        if output == 0:
            return 'Mask is weared inccorectly'
        elif output == 1:
            return 'With mask'
        elif output == 2:
            return 'Without mask'
        
if __name__ == '__main__':
    print((test('F:/Python/Projects/Mask-detection/mask_detector.pth', 'F:/Python/Projects/Random stuff/incorrect_masks111.jpg')))