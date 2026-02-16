#!/bin/bash

#EMOS
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python src/genpp/train.py --config-name base_emos data.batch_size=64 model.optimizer.lr=0.001 data=wb2_cut model/lr_scheduler=reduceLROnPlateau "logger.tags=[emos,final]"


# DRN
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_drn data.batch_size=32 "model.hidden_channels=[128, 64]" model.optimizer.lr=0.0001 data=wb2_full_minmax model/lr_scheduler=reduceLROnPlateau "logger.tags=[drn,final]"


# LNGM (fromer chen model)
# Indirect
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_noise data.batch_size=8 model.internal_td_scaling=abs model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, es, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_noise data.batch_size=8 model.internal_td_scaling=abs model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, pes, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_noise data.batch_size=8 model.internal_td_scaling=abs model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=multiscale_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, mses, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_noise data.batch_size=4 model.internal_td_scaling=abs model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=multiscale_patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, mspes, final, indirect]" trainer.accumulate_grad_batches=2

# Direct
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_direct data.batch_size=8 model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, es, final, direct]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_direct data.batch_size=8 model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, pes, final, direct]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_direct data.batch_size=8 model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=multiscale_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, mses, final, direct]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_chen model=cnn_chen_direct data.batch_size=4 model.optimizer.lr=0.001 "model.std_unet_channels=[32, 64, 128]" "model.decoder_unet_channels=[16, 32]" model/loss_fn=multiscale_patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[lngm, mspes, final, direct]" trainer.accumulate_grad_batches=2


# ENGRESSION
# Indirect
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=8 model.internal_td_scaling=learned model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, es, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=8 model.internal_td_scaling=learned model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, pes, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=8 model.internal_td_scaling=learned model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=multiscale_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, mses, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=4 model.internal_td_scaling=learned model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=multiscale_patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, mspes, final, indirect]" trainer.accumulate_grad_batches=2

# Direct
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=8 model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, es, final, direct]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=8 model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, pes, final, direct]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=8 model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=multiscale_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, mses, final, direct]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_engression model=cnn_engression_noise data.batch_size=4 model.optimizer.lr=0.001 "model.channels=[16,32]" model/loss_fn=multiscale_patchwise_energy_score data=wb2_full_pad_x model/lr_scheduler=reduceLROnPlateau "logger.tags=[engression, mspes, final, direct]" trainer.accumulate_grad_batches=2



# FMUnet
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_fm_unet model=fm_unet_noise data.batch_size=8 model.internal_td_scaling=abs model.optimizer.lr=0.0001 model.backbone.num_residual_layers=2 "model.backbone.channels=[64, 128]" data=wb2_full_pad_xy model/lr_scheduler=reduceLROnPlateau "logger.tags=[fm_unet, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_fm_unet model=fm_unet_direct data.batch_size=8 model.optimizer.lr=0.0001 model.backbone.num_residual_layers=2 "model.backbone.channels=[64, 128]" data=wb2_full_pad_xy model/lr_scheduler=reduceLROnPlateau "logger.tags=[fm_unet, final, direct]"

# FMUViT
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_fm_uvit model=fm_uvit_noise data.batch_size=8 model.internal_td_scaling=abs model.optimizer.lr=0.001 model.backbone.depth=2 model.backbone.embed_dim=128 data=wb2_full_pad_xy model/lr_scheduler=reduceLROnPlateau "logger.tags=[fm_uvit, final, indirect]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_fm_uvit model=fm_uvit_direct data.batch_size=8 model.optimizer.lr=0.001 model.backbone.depth=2 model.backbone.embed_dim=128 data=wb2_full_pad_xy model/lr_scheduler=reduceLROnPlateau "logger.tags=[fm_uvit, final, direct]"

# FMUViT_CFG
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_fm_uvit_cfg model=fm_uvit_cfg_noise data.batch_size=8 model.internal_td_scaling=abs model.optimizer.lr=0.001 model.backbone.depth=2 model.backbone.embed_dim=128 data=wb2_full_pad_xy model/lr_scheduler=reduceLROnPlateau "logger.tags=[fm_uvit_cfg, final, indirect, extra]"
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu  python src/genpp/train.py --config-name base_fm_uvit_cfg model=fm_uvit_cfg_direct data.batch_size=8 model.optimizer.lr=0.001 model.backbone.depth=2 model.backbone.embed_dim=128 data=wb2_full_pad_xy model/lr_scheduler=reduceLROnPlateau "logger.tags=[fm_uvit_cfg, final, direct, extra]"
