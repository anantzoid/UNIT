"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler, get_border_mask
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import itertools
from gan_nets import *
from helpers import _compute_true_acc, _compute_fake_acc

class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.dis = COCOSharedDis(hyperparameters['input_dim_a'], hyperparameters['input_dim_b'], hyperparameters['dis'])
        self.gen = COCOResGen2(hyperparameters['input_dim_a'], hyperparameters['input_dim_b'], hyperparameters['gen'])

        # Setup the optimizers
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        # Network weight initialization
        self.dis.apply(gaussian_weights_init)
        self.gen.apply(gaussian_weights_init)
        # Setup the loss function for training
        self.ll_loss_criterion_a = torch.nn.L1Loss()
        self.ll_loss_criterion_b = torch.nn.L1Loss()

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        if hyperparameters['border_w'] > 0:
          self.border_mask = torch.from_numpy(get_border_mask((1, 3, 256, 256), hyperparameters['border_ratio'])).float()
          self.border_mask = Variable(self.border_mask.cuda(), requires_grad=False)


        # self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        # self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        # self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        # self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        # self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # # Setup the optimizers
        # beta1 = hyperparameters['beta1']
        # beta2 = hyperparameters['beta2']
        # dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        # gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        # self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        # self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # # Network weight initialization
        # self.apply(weights_init(hyperparameters['init']))
        # self.dis_a.apply(weights_init('gaussian'))
        # self.dis_b.apply(weights_init('gaussian'))

        # # Load VGG model if needed
        # if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
        #     self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
        #     self.vgg.eval()
        #     for param in self.vgg.parameters():
        #         param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def _compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def compute_border_loss(self, img, target):
      return torch.sum((torch.abs(img - target) * self.border_mask) / torch.sum(self.border_mask))

    def gen_update(self, images_a, images_b, hyperparameters):
      self.gen.zero_grad()
      x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a, images_b)
      x_bab, shared_bab = self.gen.forward_a2b(x_ba)
      x_aba, shared_aba = self.gen.forward_b2a(x_ab)
      outs_a, outs_b = self.dis(x_ba,x_ab)
      for it, (out_a, out_b) in enumerate(itertools.izip(outs_a, outs_b)):
        outputs_a = nn.functional.sigmoid(out_a)
        outputs_b = nn.functional.sigmoid(out_b)
        all_ones = Variable(torch.ones((outputs_a.size(0))).cuda())
        if it==0:
          ad_loss_a = nn.functional.binary_cross_entropy(outputs_a, all_ones)
          ad_loss_b = nn.functional.binary_cross_entropy(outputs_b, all_ones)
        else:
          ad_loss_a += nn.functional.binary_cross_entropy(outputs_a, all_ones)
          ad_loss_b += nn.functional.binary_cross_entropy(outputs_b, all_ones)

      enc_loss  = self._compute_kl(shared)
      enc_bab_loss = self._compute_kl(shared_bab)
      enc_aba_loss = self._compute_kl(shared_aba)
      ll_loss_a = self.ll_loss_criterion_a(x_aa, images_a)
      ll_loss_b = self.ll_loss_criterion_b(x_bb, images_b)
      ll_loss_aba = self.ll_loss_criterion_a(x_aba, images_a)
      ll_loss_bab = self.ll_loss_criterion_b(x_bab, images_b)

      border_loss = self.compute_border_loss(images_b, x_ba) if hyperparameters['border_w'] > 0 else 0

      total_loss = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b) + \
                  hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b) + \
                  hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab) + \
                  hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss) + \
                  hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss) + \
                  hyperparameters['border_w'] * (border_loss)
      total_loss.backward()
      self.gen_opt.step()
      self.gen_enc_loss = enc_loss.data.cpu().numpy()[0]
      self.gen_enc_bab_loss = enc_bab_loss.data.cpu().numpy()[0]
      self.gen_enc_aba_loss = enc_aba_loss.data.cpu().numpy()[0]
      self.gen_ad_loss_a = ad_loss_a.data.cpu().numpy()[0]
      self.gen_ad_loss_b = ad_loss_b.data.cpu().numpy()[0]
      self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
      self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
      self.gen_ll_loss_aba = ll_loss_aba.data.cpu().numpy()[0]
      self.gen_ll_loss_bab = ll_loss_bab.data.cpu().numpy()[0]
      self.gen_total_loss = total_loss.data.cpu().numpy()[0]
      if hyperparameters['border_w'] > 0:
        self.border_loss = border_loss.data.cpu().numpy()[0]
      return (x_aa, x_ba, x_ab, x_bb, x_aba, x_bab)

    def dis_update(self, images_a, images_b, hyperparameters):
      self.dis.zero_grad()
      x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a, images_b)
      data_a = torch.cat((images_a, x_ba), 0)
      data_b = torch.cat((images_b, x_ab), 0)
      res_a, res_b = self.dis(data_a,data_b)
      # res_true_a, res_true_b = self.dis(images_a,images_b)
      # res_fake_a, res_fake_b = self.dis(x_ba, x_ab)
      for it, (this_a, this_b) in enumerate(itertools.izip(res_a, res_b)):
        out_a = nn.functional.sigmoid(this_a)
        out_b = nn.functional.sigmoid(this_b)
        out_true_a, out_fake_a = torch.split(out_a, out_a.size(0) // 2, 0)
        out_true_b, out_fake_b = torch.split(out_b, out_b.size(0) // 2, 0)
        out_true_n = out_true_a.size(0)
        out_fake_n = out_fake_a.size(0)
        all1 = Variable(torch.ones((out_true_n)).cuda())
        all0 = Variable(torch.zeros((out_fake_n)).cuda())
        ad_true_loss_a = nn.functional.binary_cross_entropy(out_true_a, all1)
        ad_true_loss_b = nn.functional.binary_cross_entropy(out_true_b, all1)
        ad_fake_loss_a = nn.functional.binary_cross_entropy(out_fake_a, all0)
        ad_fake_loss_b = nn.functional.binary_cross_entropy(out_fake_b, all0)
        if it==0:
          ad_loss_a = ad_true_loss_a + ad_fake_loss_a
          ad_loss_b = ad_true_loss_b + ad_fake_loss_b
        else:
          ad_loss_a += ad_true_loss_a + ad_fake_loss_a
          ad_loss_b += ad_true_loss_b + ad_fake_loss_b
        true_a_acc = _compute_true_acc(out_true_a)
        true_b_acc = _compute_true_acc(out_true_b)
        fake_a_acc = _compute_fake_acc(out_fake_a)
        fake_b_acc = _compute_fake_acc(out_fake_b)
        exec( 'self.dis_true_acc_%d = 0.5 * (true_a_acc + true_b_acc)' %it)
        exec( 'self.dis_fake_acc_%d = 0.5 * (fake_a_acc + fake_b_acc)' %it)
      loss = hyperparameters['gan_w'] * ( ad_loss_a + ad_loss_b )
      loss.backward()
      self.dis_opt.step()
      self.dis_loss = loss.data.cpu().numpy()[0]
      return

    # def gen_update(self, x_a, x_b, hyperparameters):
    #     self.gen_opt.zero_grad()
    #     # encode
    #     h_a, n_a = self.gen_a.encode(x_a)
    #     h_b, n_b = self.gen_b.encode(x_b)
    #     # decode (within domain)
    #     x_a_recon = self.gen_a.decode(h_a + n_a)
    #     x_b_recon = self.gen_b.decode(h_b + n_b)
    #     # decode (cross domain)
    #     x_ba = self.gen_a.decode(h_b + n_b)
    #     x_ab = self.gen_b.decode(h_a + n_a)
    #     # encode again
    #     h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
    #     h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
    #     # decode again (if needed)
    #     x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
    #     x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

    #     # reconstruction loss
    #     self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
    #     self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
    #     self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
    #     self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
    #     self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
    #     self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
    #     self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
    #     self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
    #     # GAN loss
    #     self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
    #     self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
    #     # domain-invariant perceptual loss
    #     self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
    #     self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
    #     # total loss
    #     self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
    #                           hyperparameters['gan_w'] * self.loss_gen_adv_b + \
    #                           hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
    #                           hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
    #                           hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
    #                           hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
    #                           hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
    #                           hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
    #                           hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
    #                           hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
    #                           hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
    #                           hyperparameters['vgg_w'] * self.loss_gen_vgg_b
    #     self.loss_gen_total.backward()
    #     self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    
    def assemble_outputs(self, images_a, images_b, network_outputs):
        images_a = self.normalize_image(images_a)
        images_b = self.normalize_image(images_b)
        x_aa = self.normalize_image(network_outputs[0])
        x_ba = self.normalize_image(network_outputs[1])
        x_ab = self.normalize_image(network_outputs[2])
        x_bb = self.normalize_image(network_outputs[3])
        x_aba = self.normalize_image(network_outputs[4])
        x_bab = self.normalize_image(network_outputs[5])
        return torch.cat((images_a[0:1, ::], x_aa[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::],
                          images_b[0:1, ::], x_bb[0:1, ::], x_ba[0:1, ::], x_bab[0:1, ::]), 3)

    def sample(self, x_a, x_b):
        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    # def dis_update(self, x_a, x_b, hyperparameters):
    #     self.dis_opt.zero_grad()
    #     # encode
    #     h_a, n_a = self.gen_a.encode(x_a)
    #     h_b, n_b = self.gen_b.encode(x_b)
    #     # decode (cross domain)
    #     x_ba = self.gen_a.decode(h_b + n_b)
    #     x_ab = self.gen_b.decode(h_a + n_a)
    #     # D loss
    #     self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
    #     self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
    #     self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
    #     self.loss_dis_total.backward()
    #     self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, snapshot_prefix, hyperparameters):
        dirname = snapshot_prefix#os.path.dirname(snapshot_prefix)
        last_model_name = get_model_list(dirname,"gen")
        if last_model_name is None:
          return 0
        self.gen.load_state_dict(torch.load(last_model_name))
        iterations = int(last_model_name[-12:-4])
        last_model_name = get_model_list(dirname, "dis")
        self.dis.load_state_dict(torch.load(last_model_name))
        # Load optimizers
        state_dict = torch.load(os.path.join(snapshot_prefix, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        # NOTE Temp workaround fix from torch/optim/lr_scheduler.py
        for group in self.dis_opt.param_groups:
          group.setdefault('initial_lr', group['lr'])
        for group in self.gen_opt.param_groups:
          group.setdefault('initial_lr', group['lr'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    # def resume(self, checkpoint_dir, hyperparameters):
    #     # Load generators
    #     last_model_name = get_model_list(checkpoint_dir, "gen")
    #     state_dict = torch.load(last_model_name)
    #     self.gen_a.load_state_dict(state_dict['a'])
    #     self.gen_b.load_state_dict(state_dict['b'])
    #     iterations = int(last_model_name[-11:-3])
    #     # Load discriminators
    #     last_model_name = get_model_list(checkpoint_dir, "dis")
    #     state_dict = torch.load(last_model_name)
    #     self.dis_a.load_state_dict(state_dict['a'])
    #     self.dis_b.load_state_dict(state_dict['b'])
    #     # Load optimizers
    #     state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
    #     self.dis_opt.load_state_dict(state_dict['dis'])
    #     self.gen_opt.load_state_dict(state_dict['gen'])
    #     # Reinitilize schedulers
    #     self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
    #     self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
    #     print('Resume from iteration %d' % iterations)
    #     return iterations

    def save(self, snapshot_prefix, iterations):
        gen_filename = os.path.join(snapshot_prefix, 'gen_%08d.pkl' % (iterations + 1))
        dis_filename = os.path.join(snapshot_prefix, 'dis_%08d.pkl' % (iterations + 1))
      
        opt_name = os.path.join(snapshot_prefix, 'optimizer.pt')
        torch.save(self.gen.state_dict(), gen_filename)
        torch.save(self.dis.state_dict(), dis_filename)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

    def normalize_image(self, x):
      return x[:,0:3,:,:]
