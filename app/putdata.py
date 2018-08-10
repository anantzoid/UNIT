import os
from PIL import Image
from skimage import io
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt


done = os.listdir("/home/anant/UNIT/app/static/humerus_t/") + os.listdir("/home/anant/UNIT/app/static/humerus/")
n = json.load(open("/data2/generative/0731humerus_testset1/gendata_metadata.json")).keys()
#for i in os.listdir("/data2/generative/0716humerus_uniq_clean/testB/"):
for i in n:
  if "_0" not in i:
    continue
  if i in done:
    continue

  i = i.split(".")[0].split("_")[0]
  im = plt.imread("/data2/fullbody_cropped_apr2018/png/png_labeled/"+i+".png")
  im = np.clip(im * 255, 0, 255).astype('uint8')
  im = Image.fromarray(im)
  im.save("/home/anant/UNIT/app/static/humerus_new/"+i+".png") 
