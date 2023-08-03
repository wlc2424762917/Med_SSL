# clear gpu
sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
sudo fuser -v /dev/nvidia1 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
sudo fuser -v /dev/nvidia2 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
sudo fuser -v /dev/nvidia3 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

CUDA_VISIBLE_DEVICES=3 nohup python main_MIM_pretrain.py --batch_size=24 --accu_grad=1 --max_epochs=24000 --val_every=1 --logdir=LGE_MIM_test0 >> log_swin_MIM_LGE_test0 &
