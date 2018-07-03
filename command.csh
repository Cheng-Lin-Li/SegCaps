python3 ./main.py --train=1 --test=0 --manip=0 --data_root_dir=data --net segcapsr3 --initial_lr 0.01 --loglevel 2 --Kfold 4 --loss dice --dataset mscoco17 --recon_wei 5 --which_gpu -1 --gpus 1

python3 ./main.py --train=1 --test=0 --manip=0 --data_root_dir=data --net capsbasic --initial_lr 0.01 --loglevel 2 --Kfold 4 --loss dice --dataset mscoco17 --recon_wei 5 --which_gpu -1 --gpus 1

#python3 ./main.py --train=1 --test=0 --manip=0 --data_root_dir=data --net unet --initial_lr 0.01 --loglevel 2 --Kfold 1 --loss w_bce --dataset mscoco17 --recon_wei 2


