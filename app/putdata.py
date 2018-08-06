import os
from PIL import Image
from skimage import io
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

for i in os.listdir("/data2/generative/0731humerus_testset1/testB"):
  i = i.split(".")[0].split("_")[0]
  im = plt.imread("/data2/fullbody_cropped_apr2018/png/png_labeled/"+i+".png")
  im = np.clip(im * 255, 0, 255).astype('uint8')
  im = Image.fromarray(im)
  im.save("/home/anant/UNIT/app/static/humerus/"+i+".png") 
