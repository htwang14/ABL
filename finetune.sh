#!/bin/bash

python backdoor_finetune.py --gpu 0 --trigger_type badnet_sq 
# python backdoor_finetune.py --gpu 0 --trigger_type badnet_grid 
# python backdoor_finetune.py --gpu 0 --trigger_type trojan_3x3
# python backdoor_finetune.py --gpu 0 --trigger_type trojan_8x8 
# python backdoor_finetune.py --gpu 0 --trigger_type trojan_wm 
# python backdoor_finetune.py --gpu 0 --trigger_type l0_inv 
# python backdoor_finetune.py --gpu 0 --trigger_type l2_inv 
# python backdoor_finetune.py --gpu 0 --trigger_type blend 
# python backdoor_finetune.py --gpu 0 --trigger_type smooth 
# python backdoor_finetune.py --gpu 0 --trigger_type sig --target_type cleanLabel 

