python main.py --seg
sudo fuser -v /dev/nvidia1 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

# edsr
python main.py --sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "edsr" --lr 0.0001 --batch_size 1 --val_every 10
CUDA_VISIBLE_DEVICES=3 python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 5
python main.py --seg_sr --data "./data/acdc_seg_sr/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 1

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --logdir "./log/res_seg_sr_btcv_0/" >> log_res_seg_sr_btcv &

CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --save_checkpoint --logdir "./log/res_seg_sr_btcv_0.1_1/" >> log_res_seg_sr_btcv_1 &

CUDA_VISIBLE_DEVICES=2 python main.py --sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "edsr" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --logdir "./log/edsr_0/"
CUDA_VISIBLE_DEVICES=3 python main.py --sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "edsr" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --logdir "./log/edsr_0/"
CUDA_VISIBLE_DEVICES=3 nohup python main.py --sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "edsr" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --logdir "./log/edsr_0/" >> log_edsr_btcv &


CUDA_VISIBLE_DEVICES=2 python main.py --seg --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "unet" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --logdir "./log/unet_0/"
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "unet" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --logdir "./log/unet_0/" >> log_unet_btcv_224 &


CUDA_VISIBLE_DEVICES=2 nohup python train_res_SR_SEG_net_btcv_sseg.py >> sseg_single_branch &

CUDA_VISIBLE_DEVICES=1 nohup python train_res_SR_SEG_net_btcv_sseg.py >> sseg_single_branch_new &


CUDA_VISIBLE_DEVICES=1 python train_res_SR_SEG_net_btcv_sseg.py

CUDA_VISIBLE_DEVICES=1 nohup python train_res_SR_SEG_net_btcv_sseg.py >> log_sseg_single_branch_btcv &

CUDA_VISIBLE_DEVICES=2 python train_origin_unet_btcv_sseg.py

python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --lr 0.0001 --batch_size 8 --val_every 1 --srloss2segloss 0

CUDA_VISIBLE_DEVICES=1 python train_res_SR_SEG_net_btcv_dual.py

python test.py --sr --exp_name "edsr_0" --data "./data/synapse_seg_sr/" --pretrained_dir "/home/lichao/Med_Img/runs/edsr_0" --pretrained_model_name "model_final.pt"
/home/lichao/Med_Img/runs/log/res_seg_sr_btcv_0dot1/model.pt
python test.py --seg_sr --exp_name "res_seg_sr_btcv_0.1" --model "res_sr_seg_net_with_skip" --data "./data/synapse_seg_sr/" --pretrained_dir "/home/lichao/Med_Img/runs/log/res_seg_sr_btcv_0dot1" --pretrained_model_name "model.pt"

sudo fuser -v /dev/nvidia1 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

CUDA_VISIBLE_DEVICES=1 nohup python train_res_SR_SEG_net_btcv_dual.py >> log_dual_btcv_new_skip_connect &

# train swintrans_sr_seg_net btcv
CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net" --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/swintrans_seg_sr_btcv_0/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_btcv_0/" >> log_swinTrans_seg_sr_btcv &
python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net" --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/swintrans_seg_sr_btcv_0/"
CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net" --lr 0.0001 --batch_size 24 --val_every 1 --logdir "./log/swintrans_seg_sr_btcv_b_24/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_btcv_b24/" >> log_swinTrans_seg_sr_btcv_b24 &

# train biformer_sr_seg_net btcv
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net" --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/biformer_seg_sr_btcv_0/" --save_checkpoint --logdir "./log/biformer_seg_sr_btcv_0/" >> log_biformer_seg_sr_btcv &
python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net" --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/swintrans_seg_sr_btcv_0/"
# train biformer_sr_seg_net btcv winsize = 2
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --data "./data/synapse_seg_sr/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net" --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/biformer_seg_sr_btcv_win2/" --save_checkpoint --logdir "./log/biformer_seg_sr_btcv_win2/" >> log_biformer_seg_sr_btcv_win2 &

# train swintrans_sr_seg_net LGE
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_skip" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/swinTrans_seg_sr_LGE_3/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_LGE_3/" >> log_swinTrans_seg_LGE_3 & # 0, 255; 64 --> 512
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_skip" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/swinTrans_seg_sr_LGE_3.1/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_LGE_3.1/" >> log_swinTrans_seg_LGE_3.1 & # 0, 255; 64 --> 512
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_full_skip" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/swinTrans_seg_sr_full_skip_LGE_1/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_full_skip_LGE_1/" >> log_swinTrans_seg_sr_full_skip_LGE_1 & # 0, 255; 64 --> 512
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_skip_fa" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/swinTrans_seg_sr_skip_fa_LGE_0/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_skip_fa_LGE_0/" >> log_swinTrans_seg_sr_skip_fa_LGE_0 & # 0, 255; 64 --> 512
CUDA_VISIBLE_DEVICES=2 nohup python train_swin_SR_SEG_net_lge_dual_fa.py >> log_swinTrans_seg_sr_skip_fa_LGE_0 & # 0, 255; 64 --> 512

# train swintrans_sr_seg_net LGE from ssl
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --model "swintrans_sr_seg_net_with_skip" --resume_ckpt --pretrained_dir "./Pretrain/runs/LGE_pre" --pretrained_model_name "model_bestValRMSE.pt" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/swinTrans_seg_sr_skip_LGE_from_ssl/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_skip_LGE_from_ssl/" >> log_swinTrans_seg_sr_skip_LGE_from_ssl & # 0, 255; 64 --> 512

# train swintrans_sr_seg_net_full_skip LGE from ssl
CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --model "swintrans_sr_seg_net_with_full_skip" --resume_ckpt --pretrained_dir "./Pretrain/runs/LGE_pre" --pretrained_model_name "model_bestValRMSE.pt" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/swinTrans_seg_sr_full_skip_LGE_from_ssl/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_full_skip_LGE_from_ssl/" >> log_swinTrans_seg_sr_full_skip_LGE_from_ssl & # 0, 255; 64 --> 512

# train swintrans_sr_seg_net DTI
CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg --scale 1 --out_channels 2 --data "./data/DTI_post_paired/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --logdir "./log/swinTrans_sr_seg_net_with_skip_DTI/" >> log_swinTrans_sr_seg_net_with_skip_DTI &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg_sr --scale 1 --out_channels 2 --data "./data/DTI_post_paired/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --srloss2segloss 0 --logdir "./log/swinTrans_sr_seg_net_with_skip_DTI/" >> log_swinTrans_sr_seg_net_with_skip_DTI &

# pretrain swintrans_sr_seg_net LGE only sr
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_skip" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 1 --segloss2srloss 0 --logdir "./log/swinTrans_seg_sr_LGE_only_sr/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_LGE_only_sr/" >> log_swinTrans_seg_LGE_only_sr & # 0, 255; 64 --> 512

# pretrain swintrans_sr_seg_net LGE only sr from ssl
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2 --val_metric "psnr"  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --model "swintrans_sr_seg_net_with_skip" --resume_ckpt --pretrained_dir "./Pretrain/runs/LGE_pre" --pretrained_model_name "model_bestValRMSE.pt" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --srloss2segloss 1 --segloss2srloss 0 --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/swinTrans_seg_sr_skip_LGE_SR_from_ssl/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_skip_LGE_SR_from_ssl/" >> log_swinTrans_seg_sr_skip_LGE_SR_from_ssl & # 0, 255; 64 --> 512
CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg_sr --scale 8 --out_channels 2 --val_metric "psnr"  --data "./data/LGE_post_m/" --json_list "dataset_0.json" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --model "swintrans_sr_seg_net_with_skip" --resume_ckpt --pretrained_dir "./Pretrain/runs/LGE_pre" --pretrained_model_name "model_bestValRMSE.pt" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --srloss2segloss 1 --segloss2srloss 0 --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/swinTrans_seg_sr_skip_LGE_SR_from_ssl_m/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_skip_LGE_SR_from_ssl_m/" >> log_swinTrans_seg_sr_skip_LGE_SR_from_ssl_m & # 0, 255; 64 --> 512

# train swintrans_sr_seg_net LGE from SR_ssl
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2 --val_metric "dice"  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --model "swintrans_sr_seg_net_with_skip" --resume_ckpt --pretrained_dir "./runs/log/swinTrans_seg_sr_skip_LGE_SR_from_ssl/" --pretrained_model_name "model.pt" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --srloss2segloss 0 --segloss2srloss 1 --lr 0.0001 --batch_size 16 --val_every 1 --logdir "./log/swinTrans_seg_sr_skip_LGE_SEG_from_SR_ssl/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_skip_LGE_SEG_from_SR_ssl/" >> log_swinTrans_seg_sr_skip_LGE_SEG_from_SR_ssl & # 0, 255; 64 --> 512

# train swintrans_sr_seg_net_FA LGE from SR_ssl
# CUDA_VISIBLE_DEVICES=3 nohup python main.py --seg_sr --scale 8 --out_channels 2 --val_metric "dice"  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net_with_skip_fa" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --resume_ckpt --pretrained_dir "./runs/log/swinTrans_seg_sr_skip_LGE_SR_from_ssl/" --pretrained_model_name "model.pt" --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --segloss2srloss 1 --logdir "./log/swinTrans_seg_sr_fa_LGE_from_SR_ssl/" --save_checkpoint --logdir "./log/swinTrans_seg_sr_fa_LGE_from_SR_ssl/" >> log_swinTrans_seg_sr_fa_LGE_from_SR_ssl & # 0, 255; 64 --> 512
CUDA_VISIBLE_DEVICES=3 nohup python train_swin_SR_SEG_net_lge_dual_fa.py >> log_swinTrans_seg_sr_fa_LGE_from_SR_ssl &



python main.py --seg_sr --out_channels 2 --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "swintrans_sr_seg_net" --lr 0.0001 --batch_size 16 --val_every 1
# train swintrans_sr_seg_net LGE only seg
CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg --scale 1 --out_channels 2 --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip_single_branch" --lr 0.0001 --batch_size 16 --val_every 1 --save_checkpoint --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 64 --out_shape_y 64 --logdir "./log/res_sr_seg_net_with_skip_single_branch_0/" >> log_res_sr_seg_net_with_skip_single_branch_0 &


# train res_sr_seg_net LGE
# CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg_sr --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/res_seg_sr_LGE_0/" --save_checkpoint --logdir "./log/res_seg_sr_LGE_0/" >> log_res_seg_LGE_0 &
# CUDA_VISIBLE_DEVICES=1 nohup python main.py --seg_sr --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/res_seg_sr_LGE_1/" --save_checkpoint --logdir "./log/res_seg_sr_LGE_1/" >> log_res_seg_LGE_1 & # wrong -175, 255; 224 --> 448
CUDA_VISIBLE_DEVICES=2 nohup python main.py --seg_sr --scale 8 --out_channels 2  --data "./data/LGE_post_p/" --json_list "dataset_0.json" --model "res_sr_seg_net_with_skip" --a_min 0 --a_max 255 --in_shape_x 64 --in_shape_y 64 --out_shape_x 512 --out_shape_y 512 --lr 0.0001 --batch_size 16 --val_every 1 --srloss2segloss 0.1 --logdir "./log/res_seg_sr_LGE_3/" --save_checkpoint --logdir "./log/res_seg_sr_LGE_3/" >> log_res_seg_LGE_3 & # 0, 255; 64 --> 512


# train res_sr_seg_net_fa BTCV
CUDA_VISIBLE_DEVICES=3 nohup python train_res_SR_SEG_net_btcv_dual_fa.py >> log_res_dual_fa_btcv &

# ssl swinTrans
CUDA_VISIBLE_DEVICES=3 nohup python main.py --batch_size=10 --num_steps=24000 --lrdecay --eval_num=1 --logdir=LGE_pre >> log_swin_ssl_LGE &
CUDA_VISIBLE_DEVICES=3 nohup python main.py --batch_size=10 --num_steps=24000 --lrdecay --eval_num=1 --logdir=LGE_pre_1 >> log_swin_ssl_LGE_1 &


CUDA_VISIBLE_DEVICES=3 nohup python main.py --batch_size=10 --num_steps=24000 --lrdecay --eval_num=1 --logdir=metadataset_pre_1 >> log_swin_ssl_metadataset_1 &


# validate ssl result
CUDA_VISIBLE_DEVICES=3 python validate.py --checkpoint ./runs/LGE_pre_1/model_bestValRMSE.pt --dataset "LGE"
CUDA_VISIBLE_DEVICES=3 python validate.py --checkpoint ./runs/LGE_pre_1/model_bestValRMSE.pt --dataset "LGE" --save_dir "./recon_results_X2"

# clear gpu
sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
sudo fuser -v /dev/nvidia1 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
sudo fuser -v /dev/nvidia2 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
sudo fuser -v /dev/nvidia3 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh


