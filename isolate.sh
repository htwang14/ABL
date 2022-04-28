#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type badnet_sq
# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type badnet_grid
# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type trojan_3x3
# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type trojan_8x8
# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type trojan_wm
# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type l0_inv
# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type l2_inv
CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type blend
# CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type smooth
CUDA_VISIBLE_DEVICES=3 python backdoor_isolation.py --trigger_type sig --target_type cleanLabel

