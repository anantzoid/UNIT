from utils import  get_train_data_loaders, get_test_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch
import os
import tqdm
import json
import shutil

import numpy as np
from skimage import transform
from PIL import Image
import torchvision

def np_norm_im(img):
  min, max = float(np.min(img)), float(np.max(img))
  img = np.clip(img, min, max)
  img = (img - min) / (max - min + 1e-5)
  return img

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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help="gpu id", default=0)
parser.add_argument('--config', type=str, default='configs/lesion.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='/data2/unit', help="outputs path")
opts = parser.parse_args()

torch.cuda.set_device(opts.gpu)

# Load experiment setting
config = get_config(opts.config)
model_path = "/data2/unit/outputs/lesion/checkpoints/"
# Setup model and data loader
trainer = UNIT_Trainer(config)
trainer.cuda()
trainer.gen.eval()

resize = torchvision.transforms.Resize((256, 256))

with open("/data2/baseline/lesion_data/patch_clf_data.json") as f:
  im_list = json.load(f)['train_negative']

src = "/data2/generative/0709humerus_ic15/augmented_orig"
#os.makedirs(os.path.join(src, "testB"))
#for i in im_list:
#  shutil.copy(i, os.path.join(src, "testB")) 

config = {
    "batch_size": 1,
    "num_workers": 4,
    "new_size": 256,
    "crop_image_height": 256,
    "crop_image_width": 256,
    "data_root": src
    }
test_loader_b = get_test_data_loaders(config, shuffle_force=True)

ckpts = [55000, 60000, 65000]
ckpts = [40000]
for i in ckpts:
  save_path = os.path.join(opts.output_path + "/evaluation_unit", 
      "lesion_%s"%(i))
  print(save_path)
  trainer.gen.load_state_dict(torch.load(os.path.join(model_path, 'gen_000%d.pkl'%i)))
 
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  for it, images_b in enumerate(test_loader_b):
    '''
    images_b = Image.open(path).convert('RGB')
    images_b = resize(images_b)
    images_b = np.transpose(images_b, (2, 0, 1))
    #images_b = transform.resize(images_b, (256, 256))
    print(images_b.shape)
    images_b = torch.from_numpy(images_b).float().unsqueeze(0)#.unsqueeze(1)
    '''
    images_b, image_path, image_size = images_b
    images_b = Variable(images_b.cuda(), volatile=True)
    image_output, _ = trainer.gen.forward_b2a(images_b)
    
    output_image = im_trans(image_output)
    output_image = np.clip(output_image * 255, 0, 255).astype('uint8')
    im = Image.fromarray(output_image)
    image_name = image_path[0].split("/")[-1]
    im.save(os.path.join(save_path, image_name))

