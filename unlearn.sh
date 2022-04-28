#!/bin/bash

python backdoor_unlearning.py --gpu 0 --trigger_type badnet_sq --isolation_model_root weight/isolation_model/CIFAR10_badnet_sq_tuning_epochs15.pth
python backdoor_unlearning.py --gpu 0 --trigger_type badnet_grid --isolation_model_root weight/isolation_model/CIFAR10_badnet_grid_tuning_epochs15.pth
python backdoor_unlearning.py --gpu 0 --trigger_type trojan_3x3 --isolation_model_root weight/isolation_model/CIFAR10_trojan_3x3_tuning_epochs15.pth
python backdoor_unlearning.py --gpu 0 --trigger_type trojan_8x8 --isolation_model_root weight/isolation_model/CIFAR10_trojan_8x8_tuning_epochs15.pth
python backdoor_unlearning.py --gpu 0 --trigger_type trojan_wm --isolation_model_root weight/isolation_model/CIFAR10_trojan_wm_tuning_epochs15.pth
# python backdoor_unlearning.py --gpu 0 --trigger_type l0_inv --isolation_model_root weight/isolation_model/CIFAR10_l0_inv_tuning_epochs15.pth
# python backdoor_unlearning.py --gpu 0 --trigger_type l2_inv --isolation_model_root weight/isolation_model/CIFAR10_l2_inv_tuning_epochs15.pth
# python backdoor_unlearning.py --gpu 0 --trigger_type blend --isolation_model_root weight/isolation_model/CIFAR10_blend_tuning_epochs15.pth
# python backdoor_unlearning.py --gpu 0 --trigger_type smooth --isolation_model_root weight/isolation_model/CIFAR10_smooth_tuning_epochs15.pth
# python backdoor_unlearning.py --gpu 0 --trigger_type sig --target_type cleanLabel --isolation_model_root weight/isolation_model/CIFAR10_sig_tuning_epochs15.pth

