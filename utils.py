import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import  autograd
import torch
#from visualize import VisdomPlotter
import os
import pdb

class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


class minibatch_discriminator(nn.Module):
    def __init__(self, num_channels, B_dim, C_dim):
        super(minibatch_discriminator, self).__init__()
        self.B_dim = B_dim
        self.C_dim =C_dim
        self.num_channels = num_channels
        T_init = torch.randn(num_channels * 4 * 4, B_dim * C_dim) * 0.1
        self.T_tensor = nn.Parameter(T_init, requires_grad=True)

    def forward(self, inp):
        inp = inp.view(-1, self.num_channels * 4 * 4)
        M = inp.mm(self.T_tensor)
        M = M.view(-1, self.B_dim, self.C_dim)

        op1 = M.unsqueeze(3)
        op2 = M.permute(1, 2, 0).unsqueeze(0)

        output = torch.sum(torch.abs(op1 - op2), 2)
        output = torch.sum(torch.exp(-output), 2)
        output = output.view(M.size(0), -1)

        output = torch.cat((inp, output), 1)

        return output


class Utils(object):

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod

    # based on:  https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def compute_GP(netD, real_data, real_embed, fake_data, LAMBDA):
        BATCH_SIZE = real_data.size(0)
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 64, 64)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates, real_embed)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Logger(object):
    def __init__(self, vis_screen):
       # self.viz = VisdomPlotter(env_name=vis_screen)
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []
        self.hist_D_epoch = []
        self.hist_G_epoch = []
        self.hist_Dx_epoch = []
        self.hist_DGx_epoch = []

    def log_iteration_wgan(self, epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss):
        print("Epoch: %d, Gen_iteration: %d, d_loss= %f, g_loss= %f, real_loss= %f, fake_loss = %f" %
              (epoch, gen_iteration, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_loss, fake_loss))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())

    def log_iteration_gan(self, epoch, iteration, d_loss, g_loss, real_score, fake_score):
        print("Epoch: %d, iteration= %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, iteration, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            fake_score.data.cpu().mean()))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(fake_score.data.cpu().mean())

    def plot_epoch(self, epoch):
        #self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        #self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.hist_D_epoch.append(np.array(self.hist_D).mean())
        self.hist_G_epoch.append(np.array(self.hist_G).mean())        
        self.hist_D = []
        self.hist_G = []
        fig_1 = plt.figure(figsize=(8, 4))
        ax_1 = fig_1.add_subplot(111)
        print(self.hist_D)
        ax_1.plot(np.arange(epoch+1), np.array(self.hist_D_epoch))
        ax_1.legend('train')
        ax_1.set_title('Discriminator')
        fig_1.savefig('/home/cuijie/mlp_cw3/models/birds/Discriminator_train.pdf')

        fig_2 = plt.figure(figsize=(8, 4))
        ax_2 = fig_2.add_subplot(111)
        print(self.hist_G)
        ax_2.plot(np.arange(epoch+1), np.array(self.hist_G_epoch))
        ax_2.legend('train')
        ax_2.set_title('Generator')
        fig_2.savefig('/home/cuijie/mlp_cw3/models/birds/Generator_train.pdf')

    def plot_epoch_w_scores(self, epoch):
        #self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        #self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        #self.viz.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
        #self.viz.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
        self.hist_D_epoch.append(np.array(self.hist_D).mean())
        self.hist_G_epoch.append(np.array(self.hist_G).mean())
        self.hist_Dx_epoch.append(np.array(self.hist_Dx).mean())
        self.hist_DGx_epoch.append(np.array(self.hist_DGx).mean())
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    # Plot the change in the Discriminator and generator error over training.
        fig_1 = plt.figure(figsize=(8, 4))
        ax_1 = fig_1.add_subplot(111)
        ax_1.plot(np.arange(epoch+1), np.array(self.hist_D_epoch))
        ax_1.legend('train')
        ax_1.set_title('Discriminator')
        fig_1.savefig('/home/cuijie/mlp_cw3/models/birds/Discriminator_train.pdf')

        fig_2 = plt.figure(figsize=(8, 4))
        ax_2 = fig_2.add_subplot(111)
        ax_2.plot(np.arange(epoch+1), np.array(self.hist_G_epoch))
        ax_2.legend('train')
        ax_2.set_title('Generator')
        fig_2.savefig('/home/cuijie/mlp_cw3/models/birds/Generator_train.pdf')

        fig_3 = plt.figure(figsize=(8, 4))
        ax_3 = fig_3.add_subplot(111)
        ax_3.plot(np.arange(epoch+1), np.array(self.hist_Dx_epoch))
        ax_3.legend('train')
        ax_3.set_title('D(X)')
        fig_3.savefig('/home/cuijie/mlp_cw3/models/birds/D(X)_train.pdf')

        fig_4 = plt.figure(figsize=(8, 4))
        ax_4 = fig_4.add_subplot(111)
        ax_4.plot(np.arange(epoch+1), np.array(self.hist_DGx_epoch))
        ax_4.legend('train')
        ax_4.set_title('D(G(X))')
        fig_4.savefig('/home/cuijie/mlp_cw3/models/birds/D(G(X))_train.pdf')

    def draw(self, right_images, fake_images,epoch):
        #self.viz.draw('generated images', fake_images.data.cpu().numpy()[:64] * 128 + 128)
        #self.viz.draw('real images', right_images.data.cpu().numpy()[:64] * 128 + 128)
        fake_images_data = fake_images.data.cpu().numpy()[:64]
        right_images_data = right_images.data.cpu().numpy()[:64]
        if np.shape(fake_images.data.cpu().numpy()[:64])[0] < 64:
            fake_images_data  = np.concatenate((fake_images.data.cpu().numpy()[:64],np.zeros((64-np.shape(fake_images.data.cpu().numpy()[:64])[0],3,64,64))),axis=0)
        if np.shape(right_images.data.cpu().numpy()[:64])[0] < 64:
            right_images_data = np.concatenate((right_images.data.cpu().numpy()[:64],np.zeros((64-np.shape(right_images.data.cpu().numpy()[:64])[0],3,64,64))),axis=0)

        fake_images_plt_1 = np.concatenate(np.split(fake_images_data,fake_images_data.shape[0],axis=0),axis=3)
        fake_images_plt = np.transpose(np.reshape(np.concatenate(np.split(fake_images_plt_1,8,axis=3),axis=2),(3,512,512)),(1,2,0))
        plt.imsave('/home/cuijie/mlp_cw3/models/birds/fake_images_'+str(epoch)+'.png',(fake_images_plt*128+128).astype(np.uint8))


        right_images_plt_1 = np.concatenate(np.split(right_images_data,right_images_data.shape[0],axis=0),axis=3)
        right_images_plt = np.transpose(np.reshape(np.concatenate(np.split(right_images_plt_1,8,axis=3),axis=2),(3,512,512)),(1,2,0))
        plt.imsave('/home/cuijie/mlp_cw3/models/birds/real_images_'+str(epoch)+'.png',(right_images_plt*128+128).astype(np.uint8))
