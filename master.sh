#! /bin/bash

#06/22  - after commit old nets with new trainer - lesion
#python train.py --config configs/lesion.yaml --output_path /data2/unit 

#0624
# continuing above experiment as losses seem to stabilize
#python train.py --config configs/lesion.yaml --output_path /data2/unit --resume 
# with only crops
#python train.py --config configs/0624humerus.yaml --output_path /data2/unit --gpu 1

#python eval.py --config configs/0628test.yaml --patch_metadata /data2/generative/0629test/gendata_metadata.json --model_path /data2/unit/outputs/lesion/checkpoints/gen_00056000.pkl

#0702
#python train.py --config configs/0702humerus_ic3.yaml
#python train.py --config configs/0702humerus_ic3_novgg.yaml --gpu 1
#python train.py --config configs/0702humerus_ic0_cc2_novgg.yaml --gpu 2

#0703
#50 samples eval on 0702humerus_ic3, 0703humerus_crop_ic0 with models 0621humerus_00060000, 0624humerus_00176500 respectively 

#0704
# in variant1 branch

#0705
#python train.py --config configs/0705humerus.yaml --output_path /data2/unit/0705/ --gpu 0
#python train.py --config configs/0705humerus_border_sameadv.yaml --output_path /data2/unit/0705/ --gpu 1 
#python train.py --config configs/0705humerus_border_lessadv.yaml --output_path /data2/unit/0705/ --gpu 2  
python train.py --config configs/0705humerus_border_sameadv_lessratio.yaml --output_path /data2/unit/0705/ --gpu 3 
