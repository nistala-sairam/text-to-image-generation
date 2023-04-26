#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:22:23 2023

@author: shashidharkanukuntla
"""


import math
from math import sqrt

import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle classes

from dalle_pytorch import DiscreteVAE

IMAGE_SIZE = 128
IMAGE_PATH = './CUB_200_2011/CUB_200_2011/images'

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.98

NUM_TOKENS = 8192
NUM_LAYERS = 2
NUM_RESNET_BLOCKS = 2
SMOOTH_L1_LOSS = False
EMB_DIM = 512
HID_DIM = 256
KL_LOSS_WEIGHT = 0

STARTING_TEMP = 1.
TEMP_MIN = 0.5
ANNEAL_RATE = 1e-6

NUM_IMAGES_SAVE = 4

ds = ImageFolder(
    IMAGE_PATH,
    T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor()
    ])
)


dl = DataLoader(ds, BATCH_SIZE, shuffle = True)

vae_params = dict(
    image_size = IMAGE_SIZE,
    num_layers = NUM_LAYERS,
    num_tokens = NUM_TOKENS,
    codebook_dim = EMB_DIM,
    hidden_dim   = HID_DIM,
    num_resnet_blocks = NUM_RESNET_BLOCKS
)

vae = DiscreteVAE(
    **vae_params,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    kl_div_loss_weight = KL_LOSS_WEIGHT
)


assert len(ds) > 0, 'folder does not contain any images'
print(f'{len(ds)} images found for training')

def save_model(path):
    save_obj = {
        'hparams': vae_params,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

# optimizer

opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

model_config = dict(
    num_tokens = NUM_TOKENS,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    num_resnet_blocks = NUM_RESNET_BLOCKS,
    kl_loss_weight = KL_LOSS_WEIGHT
)

global_step = 0
temp = STARTING_TEMP


for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(dl):
        images = images

        loss, recons = vae(
            images,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        logs = {}

        if i % 100 == 0:
            k = NUM_IMAGES_SAVE

            with torch.no_grad():
                codes = vae.get_codebook_indices(images[:k])
                hard_recons = vae.decode(codes)

            images, recons = map(lambda t: t[:k], (images, recons))
            images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
            images, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (images, recons, hard_recons))

            save_model(f'./vae.pt')

            # temperature anneal

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # lr decay

            sched.step()

        if i % 10 == 0:
            lr = sched.get_last_lr()[0]
            print(epoch, i, f'lr - {lr:6f} loss - {loss.item()}')

            logs = {
                **logs,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item(),
                'lr': lr
            }

        global_step += 1
    
save_model('./vae-final.pt')

from random import choice
from pathlib import Path

# torch

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# vision imports

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import OpenAIDiscreteVAE, DiscreteVAE, DALLE, VQGanVAE1024
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer

# helpers

def exists(val):
    return val is not None

# argument parsing

VAE_PATH = None   # './vae.pt' - will use OpenAIs pretrained VAE if not set
DALLE_PATH = None # './dalle.pt'
TAMING = False    # use VAE from taming transformers paper
IMAGE_TEXT_FOLDER = './'
TEXT_PATH = './birds/text_c10'
BPE_PATH = None
RESUME = exists(DALLE_PATH)

EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 0.5

MODEL_DIM = 512
TEXT_SEQ_LEN = 256
DEPTH = 2
HEADS = 4
DIM_HEAD = 64
REVERSIBLE = True
VOCAB_SIZE = 50000

# tokenizer

if BPE_PATH is not None:
    tokenizer = HugTokenizer(BPE_PATH)

# reconstitute vae

if RESUME:
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'

    loaded_obj = torch.load(str(dalle_path))

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']

    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    else:
        vae = OpenAIDiscreteVAE()

    dalle_params = dict(        
        **dalle_params
    )

    IMAGE_SIZE = vae.image_size

else:
    if exists(VAE_PATH):
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        vae_klass = OpenAIDiscreteVAE if not TAMING else VQGanVAE1024
        vae = vae_klass()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        num_text_tokens = VOCAB_SIZE,
        text_seq_len = TEXT_SEQ_LEN,
        dim = MODEL_DIM,
        depth = DEPTH,
        heads = HEADS,
        dim_head = DIM_HEAD,
        reversible = REVERSIBLE
    )
    
# helpers

def save_model(path):
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'weights': dalle.state_dict()
    }

    torch.save(save_obj, path)

# from nltk.tokenize import word_tokenize

def tokenize(s):
    return tokenizer.tokenize(
        s,
        TEXT_SEQ_LEN,
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

        tokenized_text = tokenize(description).squeeze(0)
        mask = tokenized_text != 0

        image_tensor = self.image_tranform(image)
        return tokenized_text, image_tensor, mask

# create dataset and dataloader

ds = TextImageDataset(
    TEXT_PATH,
    IMAGE_PATH,
    text_len = TEXT_SEQ_LEN,
    image_size = IMAGE_SIZE
)

assert len(ds) > 0, 'dataset is empty'
print(f'{len(ds)} image-text pairs found for training')

dl = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

# initialize DALL-E

dalle = DALLE(vae = vae, **dalle_params)

if RESUME:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(dalle.parameters(), lr = LEARNING_RATE)

model_config = dict(
    depth = DEPTH,
    heads = HEADS,
    dim_head = DIM_HEAD
)

# training


for epoch in range(EPOCHS):
    for i, (text, images, mask) in enumerate(dl):
        text, images, mask = map(lambda t: t, (text, images, mask))

        loss = dalle(text, images, return_loss = True)

        loss.backward()
        clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)

        opt.step()
        opt.zero_grad()

        log = {}

        if i % 10 == 0:
            print(epoch, i, f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }

        if i % 100 == 0:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            image = dalle.generate_images(
                text[:1],
                filter_thres = 0.9    # topk sampling at 0.9
            )

            save_model(f'./dalle.pt')
    
save_model(f'./dalle-final.pt')

# Generate image from Dall-E

text_sample = tokenize('this bird is tiny with green feathers').squeeze(0)
text_sample = torch.reshape(text_sample, (1,256))
output_image = dalle.generate_images(text_sample)

transform = T.ToPILImage()

img = transform(output_image[0])

img.show()


        
    
    
    
    
    