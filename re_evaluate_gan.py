import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from txt2image_dataset import Text2ImageDataset
from models.gan_factory import gan_factory
from utils import Utils, Logger
from PIL import Image
import os
from scores import inception_score, colour_score
from torchvision.models.inception import inception_v3

cuda = torch.cuda.is_available()
print("Cuda is_available:", cuda)

class Evaluator(object):
    def __init__(self, type, dataset, split, vis_screen, save_path, pre_trained_gen, pre_trained_disc, batch_size, num_workers, epochs, visualize, l1_coef, l2_coef):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f)

        if cuda:
            self.generator = torch.nn.DataParallel(gan_factory.generator_factory(type).eval().cuda())
            self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory(type).eval().cuda())

        else:
            self.generator = torch.nn.DataParallel(gan_factory.generator_factory(type).eval())
            self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory(type).eval())
        self.cls = False
        if dataset == 'birds':
            self.dataset = Text2ImageDataset(config['birds_dataset_path'], split=split)
        elif dataset == "flicker":
            self.dataset = Text2ImageDataset(config['flickr_dataset_path'], split=split)
        elif dataset == 'flowers':
            self.dataset = Text2ImageDataset(config['flowers_dataset_path'], split=split)
        else:
            print('Dataset not supported, please select either birds or flowers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = epochs

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)



        self.save_path = save_path
        self.type = type
        self.visualize = visualize
        self.pre_trained_gen = pre_trained_gen
        self.pre_trained_disc = pre_trained_disc
        if self.visualize:
            self.logger = Logger(vis_screen)

        if cuda:
            print("in cuda")
            dtype = torch.cuda.FloatTensor
        else:
            print("out cuda")
            dtype = torch.FloatTensor

        self.inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        self.inception_model.eval();


    def evaluate_gan(self):

        encod_loss = nn.MSELoss()
        iteration = 0
        cls = self.cls
        print("Evaluate Gan")
        for epoch in range(100,self.num_epochs, 10):
            criterion = nn.BCELoss()
            l2_loss = nn.MSELoss()
            l1_loss = nn.L1Loss()
            pre_trained_gen_e = self.pre_trained_gen + str(epoch) + ".pth"
            pre_trained_disc_e = self.pre_trained_disc + str(epoch) + '.pth'
            self.discriminator.load_state_dict(torch.load(pre_trained_disc_e, map_location=lambda storage, loc: storage))
            self.generator.load_state_dict(torch.load(pre_trained_gen_e, map_location=lambda storage, loc: storage))
            for i, sample in enumerate(self.data_loader):
                iteration += 1
                right_images = Variable(sample['right_images'].float())
                right_embed = Variable(sample['right_embed'].float())
                wrong_images = Variable(sample['wrong_images'].float())
                txt = sample['txt']

                if cuda:
                    right_images = right_images.cuda()
                    right_embed = right_embed.cuda()
                    wrong_images = wrong_images.cuda()

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels)
                smoothed_real_labels = Variable(smoothed_real_labels)
                fake_labels = Variable(fake_labels)
                if cuda:
                    real_labels = real_labels.cuda()
                    smoothed_real_labels = smoothed_real_labels.cuda()
                    fake_labels = fake_labels.cuda()


                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100))
                if cuda:
                    noise = noise.cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                if cls:
                    d_loss = d_loss + wrong_loss

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100))
                if cuda:
                    noise = noise.cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)


                #======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                #===========================================
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)



                e_loss = g_loss



                img_array = fake_images.detach()
                color_score_m, color_score_std = colour_score(img_array, txt)
                print(color_score_m, color_score_std)
                incep_score_m, incep_score_std = inception_score(fake_images, self.inception_model, self.batch_size)

                self.logger.log_iteration_gan(epoch,d_loss, g_loss, e_loss, real_score, fake_score, incep_score_m, incep_score_std, color_score_m)
                self.logger.draw(right_images, fake_images)



            if self.visualize:
                self.logger.plot_epoch_w_scores(epoch)
            print(epoch)
