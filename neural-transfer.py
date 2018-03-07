import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from PIL import Image
import argparse
import numpy as np
import os

use_gpu = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

def load_image(image_path, transforms=None, max_size=None, shape=None):
    image_path=os.path.join(image_path)
    image = Image.open(image_path)
    image_size = image.size

    if max_size is not None:

        image_size = image.size

        size = np.array(image_size).astype(float)
        size = max_size / size * size
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)


    if transforms is not None:
        image = transforms(image).unsqueeze(0)


    return image.type(dtype)

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19','28']
        self.vgg19 = models.vgg19(pretrained = True).features

    def forward(self, x):
        features = []

        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def main(config):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
        ])

    content = load_image(config.content, transform, max_size = config.max_size)
    style = load_image(config.style, transform, shape = [content.size(2), content.size(3)])

    target = Variable(content.clone(), requires_grad = True)
    optimizer = torch.optim.Adam([target], lr = config.lr, betas=[0.5, 0.999])

    vgg = VGGNet()
    if use_gpu:
        vgg = vgg.cuda()

    for step in range(config.total_step):

        target_features = vgg(target)
        content_features = vgg(Variable(content))
        style_features = vgg(Variable(style))

        content_loss = 0.0
        style_loss = 0.0

        for f1, f2, f3 in zip(target_features, content_features, style_features):

            content_loss += torch.mean((f1 - f2)**2)

            n, c, h, w = f1.size()

            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            style_loss += torch.mean((f1 - f3)**2) / (c * h * w) 

        loss = content_loss + style_loss * config.style_weight

        #反向求导与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % config.log_step == 0:
            print ('Step [%d/%d], Content Loss: %.4f, Style Loss: %.4f' 
                   %(step+1, config.total_step, content_loss.data[0], style_loss.data[0]))

        if (step+1) % config.sample_step == 0:
            # Save the generated image
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().cpu().squeeze()
            img = denorm(img.data).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-%d.png' %(step+1))
            
            
class Config:
    content='drive/fish.jpg'
    style='drive/sky.jpg'
    max_size=400
    total_step=2100
    log_step=10
    sample_step=2100
    style_weight=100
    lr=0.003        

config=Config()
main(config)
