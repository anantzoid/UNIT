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
# These experiment were ran on branch: `local`
#python train.py --config configs/0702humerus_ic3.yaml
#python train.py --config configs/0702humerus_ic3_novgg.yaml --gpu 1
#python train.py --config configs/0702humerus_ic0_cc2_novgg.yaml --gpu 2

#0703
#50 samples eval on 0702humerus_ic3, 0703humerus_crop_ic0 with models 0621humerus_00060000, 0624humerus_00176500 respectively 


#0709
# epoch-wise qual analysis
#python compare_samples.py --gpu 1 --model_dir /data2/unit/0705/outputs/0705humerus --epoch_interval 1000 --epoch_limit 41000,50000  --samples 10,15
#python compare_samples.py --gpu 1 --model_dir /data2/unit/0705/outputs/0705humerus --epoch_interval 1000 --epoch_limit 51000,60000  --samples 10,20
#python compare_samples.py --gpu 1 --model_dir /data2/unit/outputs/lesion --epoch_interval 500 --epoch_limit 57000,60000  --samples 15,25 


#python eval.py --gpu 2 --config /data2/unit/outputs/lesion/config.yaml --model_path /data2/unit/outputs/lesion/checkpoints/gen_00040000.pkl
#gan_w is 1
#python train.py --config configs/0709humerus.yaml --output_path /data2/unit/ --gpu 0

#0712
#eval 
#python eval.py --gpu 2 --config /data2/unit/0705/outputs/0705humerus_border_lessadv/config.yaml --model_path /data2/unit/0705/outputs/0705humerus_border_lessadv/checkpoints/gen_00170000.pkl

#0718
#gen merged/blended image
#python eval.py --gpu 3 --config /data2/unit/outputs/0709humerus/config.yaml --model_path /data2/unit/outputs/lesion_0621humerus/checkpoints/gen_00040000.pkl

#python eval.py --gpu 3 --config /data2/unit/outputs/0709humerus/config.yaml --model_path /data2/unit/outputs/0709humerus_merged/checkpoints/gen_00035000.pkl --patch_compare

# other exps in merged2 branch

#0730
#python train.py --config configs/0730_femur.yaml --output_path /data2/unit/ --gpu 0
