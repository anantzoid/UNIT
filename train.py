"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_train_data_loaders, get_test_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images
import argparse
from torch.autograd import Variable
from new_trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torchvision
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help="gpu id", default=0)
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='/data2/unit', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

torch.cuda.set_device(opts.gpu)

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = '/data2/vgg_model'

# Setup model and data loader
trainer = UNIT_Trainer(config)
trainer.cuda()
train_loader_a, train_loader_b = get_train_data_loaders(config)
#test_loader_a, test_loader_b = get_test_data_loaders(config)
'''
train_display_images_a = Variable(torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda(), volatile=True)
train_display_images_b = Variable(torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda(), volatile=True)
test_display_images_a = Variable(torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda(), volatile=True)
test_display_images_b = Variable(torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda(), volatile=True)
'''

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a, images_b = Variable(images_a.cuda()), Variable(images_b.cuda())

        # Main training code
        trainer.dis_update(images_a, images_b, config)
        trainer.gen_update(images_a, images_b, config)


        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            # Test set images
            assembled_images = trainer.sample(images_a, images_b)
            img_filename = '%s/gen_%08d.jpg' % (image_directory, iterations + 1)
            torchvision.utils.save_image(assembled_images.data.cpu().squeeze(1), img_filename, nrow=1, normalize=True)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

