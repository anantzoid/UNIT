#! /bin/bash

#06/22  - after commit old nets with new trainer - lesion
#python train.py --config configs/lesion.yaml --output_path /data2/unit 

#0624
# continuing above experiment as losses seem to stabilize
#python train.py --config configs/lesion.yaml --output_path /data2/unit --resume 
# with only crops
#python train.py --config configs/0624humerus.yaml --output_path /data2/unit --gpu 1

python eval.py --config configs/0628test.yaml --patch_metadata /data2/generative/0629test/gendata_metadata.json --model_path /data2/unit/outputs/lesion/checkpoints/gen_00056000.pkl

#0702
# Eval on crop (the data is re-generated to account for metadata)
#python eval.py --config configs/ --patch_metadata  --model_path /data2/unit/outputs/0624humerus/checkpoints/gen_00176500.pkl

