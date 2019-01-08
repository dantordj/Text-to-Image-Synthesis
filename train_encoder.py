import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from txt2image_dataset import Text2ImageDataset
from models.gan_factory import gan_factory
from PIL import Image
import os

cuda = torch.cuda.is_available()
print("Cuda is_available:", cuda)

with open('config.yaml', 'r') as f:
        config = yaml.load(f)

def load_my_state_dict(network, state_dict):

    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

lr = 0.001
num_epochs = 20
batch_size = 64
num_workers = 8
iteration = 0
noise_dim = 100
pre_trained_gen = "gen.pth"
pre_trained_disc = "disc_100.pth"
typ = "gan"
discriminator = gan_factory.discriminator_factory(typ)
state_dict = torch.load(pre_trained_disc, map_location=lambda storage, loc: storage)
print(type(state_dict))
state_new = {}
for key, value in state_dict.items():
    state_new[key[7:]] = value
discriminator.load_state_dict(state_new)
encoder = gan_factory.encoder_factory(typ)


load_my_state_dict(encoder, state_new)
for layer in encoder.NetE_1.children():
    layer.requires_grad = False
if cuda:
    encoder = torch.nn.DataParallel(encoder.cuda())
    generator = torch.nn.DataParallel(gan_factory.generator_factory(typ).cuda())
else:
    encoder = torch.nn.DataParallel(encoder)
    generator = torch.nn.DataParallel(gan_factory.generator_factory(typ))
generator.load_state_dict(torch.load(pre_trained_gen, map_location=lambda storage, loc: storage))
dataset = Text2ImageDataset(config['birds_dataset_path'], split=0)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)
optimE = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
checkpoints_path = 'checkpoints'

save_path = ""

#generator.eval()
l2_loss = nn.MSELoss()
losses = []
for epoch in range(num_epochs):
    for sample in data_loader:
        iteration += 1
        right_images = sample["right_images"]
        right_images = Variable(right_images.float())
        right_embed = Variable(sample['right_embed'].float())
        noise = Variable(torch.randn(right_images.size(0), noise_dim), volatile=True)
        noise = noise.view(noise.size(0), 100, 1, 1)

        fake_images = Variable(generator(right_embed, noise).data)
        if cuda:

            right_embed = right_embed.cuda()
            fake_images = fake_images.cuda()

        fake_noises = encoder(fake_images.detach(), right_embed)
        if cuda:
            fake_noises.cuda()
        e_loss = l2_loss(fake_noises, noise)
        e_loss.backward()
        optimE.step()
        losses += [e_loss]
        print(e_loss)
