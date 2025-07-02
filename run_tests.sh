#!/bin/bash
python test_mvtec.py \
  --config_path config/can.yaml \
  --checkpoint_path results/can/checkpoints_15.pkl \
  --mvtec_root mvtec_ad_2 \
  --objects can

python test_mvtec.py \
  --config_path config/fabric.yaml \
  --checkpoint_path results/fabric/checkpoints_52.pkl \
  --mvtec_root mvtec_ad_2 \
  --objects fabric

python test_mvtec.py \
  --config_path config/fruit_jelly.yaml \
  --checkpoint_path results/fruit_jelly/checkpoints_12.pkl\
  --mvtec_root mvtec_ad_2 \
  --objects fruit_jelly

python test_mvtec.py \
  --config_path config/rice.yaml \
  --checkpoint_path results/rice/checkpoints_2.pkl \
  --mvtec_root mvtec_ad_2 \
  --objects rice

# 指令一：sheet_metal
python test_mvtec.py \
  --config_path config/sheet_metal.yaml \
  --checkpoint_path results/sheet_metal/checkpoints_1.pkl \
  --mvtec_root mvtec_ad_2 \
  --objects sheet_metal

# 指令二：vial
python test_mvtec.py \
  --config_path config/vial.yaml \
  --checkpoint_path results/vial/checkpoints_1.pkl \
  --mvtec_root mvtec_ad_2 \
  --objects vial

# 指令三：wallplugs
python test_mvtec.py \
  --config_path config/wallplugs.yaml \
  --checkpoint_path results/wallplugs/checkpoints_2.pkl \
  --mvtec_root mvtec_ad_2 \
  --objects wallplugs

# 指令四：walnuts
python test_mvtec.py \
  --config_path config/walnuts.yaml \
  --checkpoint_path results/walnuts/checkpoints_1.pkl \
  --mvtec_root mvtec_ad_2 \
  --objects walnuts
