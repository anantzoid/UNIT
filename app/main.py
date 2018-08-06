import matplotlib
matplotlib.use('agg')

from flask import Flask
from flask import render_template
from flask import send_from_directory
from flask import request
from flask import jsonify
from flask import session

from PIL import Image
import numpy as np
import os

import torch
from torch.autograd import Variable
import torchvision.transforms.functional as F
from torchvision import transforms

import sys
sys.path.append('../')
from trainer import UNIT_Trainer
from utils import get_config

import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8znxec]/'

def im_trans(image_output):
  image_output = np.mean(image_output.numpy(), 0)*0.5 + 0.5
  return image_output

def getmask():
  mesh = np.linspace(-1,1,256)
  x, y = np.meshgrid(mesh, mesh)
  constant = 0.5 * np.pi
  mask = np.cos(constant * x) * np.cos(constant * y)
  alpha = 1-mask
  return alpha

alpha = getmask()
img_path = "static/humerus"
dst_path = "static/humerus_t"
src_ims = [os.path.join(img_path, i) for i in os.listdir(img_path)]

@app.route("/")
def index():
  if 'im_counter' not in session:
    session['im_counter'] = -1

  session['im_counter'] += 1
  img_path = src_ims[session['im_counter']]
  return render_template('index.html', img_path=img_path)

@app.route("/getimg")
def getimg():
  session['im_counter'] += 1
  img_path = src_ims[session['im_counter']]
  return jsonify({"img": img_path})

@app.route("/coords")
def getCoords():
  coords = {
    'sx': int(request.args.get('x')),
    'sy': int(request.args.get('y'))
    }
  coords['ex'] = int(coords['sx']) + 256
  coords['ey'] = int(coords['sy']) + 256
  full_im = Image.open(src_ims[session['im_counter']]).convert('RGB')

  mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
  transform_list = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
  full_im = transform_list(full_im)
  patch = full_im[:, coords['sy']:coords['ey'], coords['sx']:coords['ex']]

  image_input = Variable(patch.unsqueeze(0).cuda(), volatile=True)
  image_output, _ = trainer.gen.forward_b2a(image_input)

  blended = (1-alpha)*image_output[0].data.cpu() + alpha*patch

  full_im[:, coords['sy']:coords['ey'], coords['sx']:coords['ex']] = blended 
  image_output_ = im_trans(full_im)

  im = np.clip(image_output_ * 255, 0, 255).astype('uint8')
  im = Image.fromarray(im)

  dicom = os.path.split(src_ims[session['im_counter']])[-1]
  im_t = os.path.join(dst_path, "%s.png"%(dicom))
  im.save(im_t)
  return jsonify({'status': 1, 'img': im_t})

if __name__ == "__main__":
  config = get_config('/home/anant/UNIT/configs/0801_humerus.yaml')
  trainer = UNIT_Trainer(config)
  trainer.gen.load_state_dict(torch.load("/data2/unit/outputs/0801_humerus/checkpoints/gen_00050000.pkl"))
  trainer.cuda()
  trainer.gen.eval()

  app.run(host='0.0.0.0', debug=True)
