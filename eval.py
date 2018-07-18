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
import torchvision.transforms.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import json
import tqdm
from datetime import datetime

import numpy as np
from skimage import transform
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help="gpu id", default=0)
parser.add_argument('--config', type=str, default='configs/lesion.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='/data2/unit', help="outputs path")
parser.add_argument('--model_path', type=str, default='', help="model path")

parser.add_argument('--debug', action="store_true")
opts = parser.parse_args()

if opts.model_path == '':
  assert "Need model path"

breakpoint = 100

torch.cuda.set_device(opts.gpu)

# Load experiment setting
config = get_config(opts.config)
patch_metadata = json.load(open(os.path.join(config['data_root'], 'gendata_metadata.json'))) 
#config['data_root'] = opts.data_root

# Setup model and data loader
trainer = UNIT_Trainer(config)
trainer.gen.load_state_dict(torch.load(opts.model_path))
trainer.cuda()
trainer.gen.eval()

test_loader_b = get_test_data_loaders(config, shuffle_force=True)
# Setup logger and output folders
base_path = opts.model_path.split("/")
model_name = base_path[-3]#os.path.splitext(os.path.basename(opts.config))[0]
ts = str(datetime.now()).split(".")[0].replace(" ", "_")
output_directory = os.path.join(opts.output_path + "/evaluation_unit", "%s_%s_%s_%s"%(config['data_root'].rstrip("/").split("/")[-1],
                            model_name, base_path[-1].split(".")[0].split("_")[-1],
                            ts))
print("output dir:", output_directory)

if not os.path.exists(output_directory):
  os.makedirs(output_directory)

for d in ['gen', 'original', 'debug', 'blended']:
  d = os.path.join(output_directory, d)
  if not os.path.exists(d):
    os.makedirs(d)

def np_norm_im(img):
  return img*0.5 + 0.5
  min, max = float(np.min(img)), float(np.max(img))
  img = np.clip(img, min, max)
  img = (img - min) / (max - min + 1e-5)
  return img

def resize_image_by_crop(im, coords, target_size):
  im = transform.resize(im, target_size)
  w, h = coords['ex'] - coords['sx'], coords['ey'] - coords['sy']
  new_w, new_h = int((256 * im.shape[1]) / w), int((256 * im.shape[0]) / h)
  new_coords = {
      'sx': int((256 * coords['sx']) / w),
      'ex': int((256 * coords['ex']) / w),
      'sy': int((256 * coords['sy']) / h),
      'ey': int((256 * coords['ey']) / h)
      }
  return transform.resize(im, (new_h, new_w)), new_coords

def im_trans(image_output):
  try:
    image_output = image_output.data
  except:
    pass
  try:
    image_output = image_output.cpu()
  except:
    pass
  image_output = np.mean(image_output[0].numpy(), 0)
  image_output = np_norm_im(image_output)#, minmax[0], minmax[1])
  return image_output

def save_merged_image(resized_image, coords, patch, save_dir):
  gen_image = np.copy(resized_image)
  gen_image[coords['sy']:coords['ey'], coords['sx']: coords['ex']] = patch
  gen_image_ = np.clip(gen_image * 255, 0, 255).astype('uint8')
  im = Image.fromarray(gen_image_)
  im.save(save_dir)
  return gen_image

def getmask():
  mesh = np.linspace(-1,1,256)
  x, y = np.meshgrid(mesh, mesh)
  constant = 0.5 * np.pi
  mask = np.cos(constant * x) * np.cos(constant * y)
  alpha = 1-mask
  return alpha
alpha = getmask()

mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
for it, images_b in tqdm.tqdm(enumerate(test_loader_b), total=breakpoint):
  images_b, image_path, image_size = images_b
  # test images are unnormalized, in range 0-1. Normalise them before translation
  images_b_ = np.mean(np.copy(images_b.numpy())[0], 0)
  images_b = Variable(F.normalize(images_b[0], mean, std).unsqueeze(0).cuda(), volatile=True)
  image_output, _ = trainer.gen.forward_b2a(images_b)

  image_name = image_path[0].split("/")[-1]
  
  dicomId = image_name.split(".")[0]
  image_name_original = "%s.png"%dicomId.split("_")[0]
  # NOTE PIL reader leads to a contrast difference. pyplot seems to work fine.
  #orig_image = np.array(default_loader(os.path.join(config['img_dir'], image_name_original), True))
  orig_image = plt.imread(os.path.join(config['img_dir'], image_name_original))

  if "_" not in dicomId:
    dicomId = "%s_0"%dicomId
  coords = patch_metadata[dicomId]['patch']
  target_size = patch_metadata[dicomId]['target_size']
 
  resized_image, coords = resize_image_by_crop(orig_image, coords, target_size) 
  
  image_output_ = im_trans(image_output)
  #images_b_ = im_trans(images_b)
  #resized_image = np_norm_im(resized_image)

  blended = (1-alpha)*image_output_ + alpha*images_b_

  # Temporary skip over a bug that arises from miscalcualted coordinates
  try:
    save_merged_image(resized_image, coords, image_output_, os.path.join(output_directory, 'gen', image_name))
  except Exception as e:
    print(str(e))
    print(images_b.size())
    print(image_output.size())
    print(image_output_.shape)
    print(coords)
    print(gen_image.shape)
    continue

  gen_images_b_ = save_merged_image(resized_image, coords, images_b_, os.path.join(output_directory, 'original', image_name))
  gen_blended = save_merged_image(resized_image, coords, blended, os.path.join(output_directory, 'blended', image_name))
  compare = np.concatenate([gen_images_b_, np.ones((gen_images_b_.shape[0],10)), gen_blended], 1)

  compare = np.clip(compare * 255, 0, 255).astype('uint8')
  im = Image.fromarray(compare)
  im.save(os.path.join(output_directory, 'debug', image_name))

  if opts.debug:
    # Save original images with patches drawn over it (debug mode)
    fig, axs = plt.subplots(1)
    axs.imshow(orig_image, cmap='gray')

    # computing respective details for creating patch
    w, h = coords['ex'] - coords['sx'], coords['ey'] - coords['sy']
    ll_x, ll_y = coords['sx'], coords['sy']
    axs.add_patch(matplotlib.patches.Rectangle((ll_x, ll_y), w, h, linewidth=1, edgecolor='r',
                    facecolor='none'))

    plt.savefig(os.path.join(output_directory, 'debug', image_name_original))
    plt.close()

  
  if it >= breakpoint:
    break
