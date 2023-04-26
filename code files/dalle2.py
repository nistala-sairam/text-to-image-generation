#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:11:53 2023

@author: shashidharkanukuntla
"""

import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from dalle2_pytorch.tokenizer import tokenizer

from pathlib import Path
from random import choice


clip = OpenAIClipAdapter()

def tokenize(s):
    return tokenizer.tokenize(
        s,
        256,
        truncate_text=True).squeeze(0)

class TextImageDataset(Dataset):
    def __init__(self, textfolder,imagefolder, text_len = 256, image_size = 128):
        super().__init__()
        textpath = Path(textfolder)
        text_files = [*textpath.glob('**/*.txt')]
        
        imagepath = Path(imagefolder)

        image_files = [
            *imagepath.glob('**/*.png'),
            *imagepath.glob('**/*.jpg'),
            *imagepath.glob('**/*.jpeg')
        ]

        text_files = {t.stem: t for t in text_files}
        image_files = {i.stem: i for i in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        self.image_tranform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size, scale = (0.75, 1.), ratio = (1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        image = Image.open(image_file)
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = choice(descriptions)
        
        # print('description: ', description);

        tokenized_text = tokenize(description).squeeze(0)
        mask = tokenized_text != 0

        image_tensor = self.image_tranform(image)
        return tokenized_text, image_tensor, mask
    
    
ds = TextImageDataset(
    './birds/text_c10',
    './CUB_200_2011/CUB_200_2011/images',
    text_len = 256,
    image_size = 256
)

dl = DataLoader(ds, batch_size = 4, shuffle = True, drop_last = True)

EPOCHS = 10

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
)

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 100,
    cond_drop_prob = 0.2
)


for epoch in range(EPOCHS):
    for i, (text, images, mask) in enumerate(dl):
        text, images, mask = map(lambda t: t, (text, images, mask))
        loss = diffusion_prior(text, images)
        loss.backward()

        log = {}
        print(epoch, i, f'loss - {loss.item()}')
        
torch.save(diffusion_prior.state_dict(), './diffusionprior.pt')
        
unet = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    text_embed_dim = 512,
    cond_on_text_encodings = True
)
    
decoder = Decoder(
    unet = unet,
    clip = clip,
    timesteps = 100,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
)

from torch.optim import Adam

LEARNING_RATE = 3e-4

opt = Adam(decoder.parameters(), lr = LEARNING_RATE)

EPOCHS = 10

for epoch in range(EPOCHS):
    for i, (text, images, mask) in enumerate(dl):
        text, images, mask = map(lambda t: t, (text, images, mask))
        loss = decoder(images, text = text)
        loss.backward()
        
        opt.step()
        opt.zero_grad()

        log = {}
        print(epoch, i, f'loss - {loss.item()}')
        
torch.save(decoder.state_dict(), './decodermodel.pt')

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)


# Generate images from Dall-E2
texts = ['this colorful bird has a yellow breast']
images = dalle2(texts)

transform = T.ToPILImage()

img = transform(images[0])

img.show()










