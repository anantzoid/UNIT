import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils import  get_train_data_loaders, get_test_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torchvision
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import json

from data import default_loader
import numpy as np
from skimage import transform

import scipy.misc
#exit()
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help="gpu id", default=0)
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='/data2/unit', help="outputs path")
parser.add_argument('--model_path', type=str, default='', help="model path")
parser.add_argument('--patch_metadata', type=str, default='', help="")

parser.add_argument('--debug', action="store_true")
opts = parser.parse_args()

if opts.model_path == '':
  assert "Need model path"

torch.cuda.set_device(opts.gpu)

# Load experiment setting
config = get_config(opts.config)
patch_metadata = json.load(open(opts.patch_metadata)) 
config['data_root'] = opts.data_root

# Setup model and data loader
trainer = UNIT_Trainer(config)
trainer.gen.load_state_dict(torch.load(opts.model_path))
trainer.cuda()
trainer.gen.eval()

test_loader_b = get_test_data_loaders(config)
# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/eval_gens", "%s_%s"%(model_name, opts.model_path.split("/")[-1].split(".")[0].split("_")[-1]))
print("output dir:", output_directory)

if not os.path.exists(output_directory):
  os.makedirs(output_directory)

if opts.debug:
  debug_savedir = os.path.join(output_directory, "debug")
  if not os.path.exists(debug_savedir):
    os.makedirs(debug_savedir)

def np_norm_im(img):
  min, max = float(np.min(img)), float(np.max(img))
  img = np.clip(img, min, max)
  img = (img - min) / (max - min + 1e-5)
  return img

def resize_image_by_crop(im, coords):
  w, h = coords['ex'] - coords['sx'], coords['ey'] - coords['sy']
  new_w, new_h = int((256 * im.shape[1]) / w), int((256 * im.shape[0]) / h)
  new_coords = {
      'sx': int((256 * coords['sx']) / w),
      'ex': int((256 * coords['ex']) / w),
      'sy': int((256 * coords['sy']) / h),
      'ey': int((256 * coords['ey']) / h)
      }
  return transform.resize(im, (new_h, new_w)), new_coords

def patch_convert(image_output):
  image_output = image_output.data.cpu()
  image_output = image_output[0,0,:,:].numpy()
  image_output = np_norm_im(image_output)#, minmax[0], minmax[1])
  return image_output

for it, images_b in enumerate(test_loader_b):
  images_b, image_path, image_size = images_b
  images_b = Variable(images_b.cuda(), volatile=True)
  image_output, _ = trainer.gen.forward_b2a(images_b)

  image_name = image_path[0].split("/")[-1]
  
  dicomId = image_name.split(".")[0]
  image_name_original = "%s.png"%dicomId
  orig_image = np.array(default_loader(os.path.join(config['img_dir'], image_name_original), True))

  if "_" not in dicomId:
    dicomId = "%s_0"%dicomId
  coords = patch_metadata[dicomId]['patch']
 
  resized_image, coords = resize_image_by_crop(orig_image, coords) 
  resized_image = np_norm_im(resized_image)
  gen_image = np.copy(resized_image)
  image_output = patch_convert(image_output)
  gen_image[coords['sy']:coords['ey'], coords['sx']: coords['ex']] = image_output

  ## SAVE image
  #normalize scales of image
  img_filename = os.path.join(output_directory, image_name)
  scipy.imsave(img_filename, gen_image)

  if opts.debug:
    dicomId = image_name.split(".")[0]
    image_name_original = "%s.png"%dicomId
    orig_image = np.array(default_loader(os.path.join(config['img_dir'], image_name_original), True))
    if "_" not in dicomId:
      dicomId = "%s_0"%dicomId
    coords = patch_metadata[dicomId]['patch']

    # Save original images with patches drawn over it (debug mode)
    fig, axs = plt.subplots(1)
    axs.imshow(orig_image, cmap='gray')

    # computing respective details for creating patch
    w, h = coords['ex'] - coords['sx'], coords['ey'] - coords['sy']
    ll_x, ll_y = coords['sx'], coords['sy']
    axs.add_patch(matplotlib.patches.Rectangle((ll_x, ll_y), w, h, linewidth=1, edgecolor='r',
                    facecolor='none'))

    plt.savefig(os.path.join(debug_savedir, image_name_original))
    plt.close()


