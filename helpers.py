import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
import matplotlib.pyplot as plt

#%% classifier for printing topk classes
class classifier:
    def __init__(self, model, categories='imagenet'):
        self.model = model
        if categories == 'imagenet':
            self.categ = _IMAGENET_CATEGORIES
        else:
            raise ValueError('Unknown Categories Specified')

    def __call__(self, I, k=1):
        tk = self.model(I).topk(k)
        for i in range(k):
            print(str(i) + ': ' +
                  self.categ[tk[1][0][i]] + 
                  ', probability: '+
                  str(tk[0][0][i].item()*100.) + '%')

def read_image(path, im_shape=[256,256], crop = True):
    I = torchvision.io.read_image(path)/255.
    I = I[None,...]

    if crop:
        l = min(I.shape[-1], I.shape[-2])
        a_2 = (I.shape[-1] - l)//2
        a_1 = (I.shape[-2] - l)//2
        I = I[..., a_1:-(a_1 + 1), a_2:-(a_2 + 1)]
        
    if im_shape is not None:
        I = transforms.Resize(tuple(im_shape), antialias=True)(I)
    return I

def show_image(I, k=0):
    plt.imshow(I[k, ...].permute(1,2,0))

def load_model(name='alexnet', **kwargs):
    if name == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        model.eval()
    elif name == 'ResNet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
    else:
        raise ValueError('Unkown model name +' )
    
    return nn.Sequential(
        model,
        nn.Softmax(dim=-1)
    )