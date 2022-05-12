#!/bin/bash 

python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type clean 
python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type clean 
python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type clean 

python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type badnet_sq
python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type badnet_sq 
python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type badnet_sq 

python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type badnet_grid
python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type badnet_grid 
python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type badnet_grid 

python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_3x3
python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_3x3
python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_3x3

python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_8x8
python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_8x8 
python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_8x8 

python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_wm
python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_wm 
python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type trojan_wm 

# python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type l0_inv
# python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type l0_inv 
# python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type l0_inv 

# python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type l2_inv
# python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type l2_inv 
# python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type l2_inv 

# python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type blend
# python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type blend 
# python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type blend 

# python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type smooth
# python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type smooth 
# python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type smooth 

# python backdoor_isolation.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type sig --target_type cleanLabel
# python backdoor_finetune.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type sig --target_type cleanLabel 
# python backdoor_unlearning.py --gpu 2 --dataset GTSRB --num_class 43 --trigger_type sig --target_type cleanLabel 
