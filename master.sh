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
# Eval on crop (the data is re-generated to account for metadata)
#python eval.py --config configs/ --patch_metadata  --model_path /data2/unit/outputs/0624humerus/checkpoints/gen_00176500.pkl


#######
# Experiments on Modified model
# Fashion of modifications: inherit architecture changes from MUNIT
# Account for change of style in generated patch with border loss
# Use additional vgg loss
#0704
# all settings as old except border loss; for direct comparision
#python train.py --config configs/0704humerus_border.yaml --gpu 0

# border, reflection padding, kaiming init in G, relu in G
#python train.py --config configs/0704humerus_border_init_pad_act.yaml --gpu 1

#zero padding
#python train.py --config configs/0704humerus_border_init_act.yaml --gpu 2
#all changes w/o vgg
#python train.py --config configs/0704humerus_border_init_pad_act_up.yaml --gpu 3






