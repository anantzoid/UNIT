from utils import get_test_data_loaders, get_config
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch
import torchvision
import os

import numpy as np
from skimage import transform
from PIL import Image, ImageDraw, ImageFont

def draw_text(text):
  img = Image.new('RGB', (256, 256), color = (255, 255, 255))
  d = ImageDraw.Draw(img)
  font = ImageFont.truetype("/data2/unit/Roboto-Black.ttf", 50)
  d.text((100,120), str(text), fill=(0,0,0), font=font)
  return torch.from_numpy((np.array(img)/255.0).transpose(2, 0, 1))

def generate_samples(opts):
  exp_name = opts.model_dir.rstrip("/").split("/")[-1]
  config = get_config(os.path.join(opts.model_dir, 'config.yaml'))
  im_save_path = '/data2/unit/evaluation_unit/epoch_compare/%s_%d_%d.png'%( 
              exp_name, opts.epoch_limit[0], opts.epoch_limit[1])
  print(im_save_path)
  test_loader_b = get_test_data_loaders(config)

  data_samples = [Variable(test_loader_b.dataset[i][0].cuda(), volatile=True) for i in range(opts.samples[0], opts.samples[1], )]
  epochs = [i for i in range(opts.epoch_limit[0], opts.epoch_limit[1]+1, opts.epoch_interval)] 
  all_gens = [torch.stack([i.data.cpu() for i in data_samples])]

  trainer = UNIT_Trainer(config)
  trainer.cuda()
  trainer.gen.eval()

  for ep in epochs:
    ep = str(ep)
    model_path = os.path.join(opts.model_dir, 'checkpoints', 'gen_%s%s.pkl'%("0"*(8-len(ep)), ep))
    print(model_path)
    # Setup model and data loader
    trainer.gen.load_state_dict(torch.load(model_path))

    epoch_gen = []
    for images_b in data_samples:
      image_output, _ = trainer.gen.forward_b2a(images_b.unsqueeze(0))
      epoch_gen.append(image_output[0].data.cpu())

    all_gens.append(torch.stack(epoch_gen))

  all_gens = torch.stack(all_gens).transpose(0, 1)#.contiguous().view(-1, 3, 256, 256)
  labels = torch.stack([draw_text("GT")] + [draw_text(i) for i in epochs]).float()

  all_gens = torch.cat([all_gens, labels.unsqueeze(0)])
  all_gens = all_gens.contiguous().view(-1, 3, 256, 256) 
  return all_gens, len(epochs)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, help="gpu id", default=0)
  parser.add_argument('--model_dir', type=str, help="path to models", default="")
  parser.add_argument('--epoch_limit', type=str, default='10000,50000')
  parser.add_argument('--epoch_interval', type=int, default=500, help='At what interval of epochs to \
        load samples.')
  parser.add_argument('--samples', type=str, default='0,10', help='Number of data samples to eval')

  opts = parser.parse_args()
  opts.epoch_limit = [int(i) for i in opts.epoch_limit.split(",")]
  opts.samples = [int(i) for i in opts.samples.split(",")]
  if len(opts.samples) == 2:
    opts.samples.append(1)

  assert len(opts.samples) == 3
  assert len(opts.epoch_limit) == 2

  torch.cuda.set_device(opts.gpu)
  all_gens, len_epochs = generate_samples(opts)
  torchvision.utils.save_image(all_gens, im_save_path, nrow=len_epochs+1,
              normalize=True, pad_value=1.0)
