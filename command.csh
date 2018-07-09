python3 ./main.py --train --data_root_dir=data --net segcapsr3 --initial_lr 0.01 --loglevel 2 --Kfold 4 --loss dice --dataset mscoco17 --recon_wei 20 --which_gpu -1 --gpus 1 --aug_data 0 

python3 ./main.py --train --data_root_dir=data --net capsbasic --initial_lr 0.01 --loglevel 2 --Kfold 4 --loss dice --dataset mscoco17 --recon_wei 20 --which_gpu -1 --gpus 1 --aug_data 0

#python3 ./main.py --train --data_root_dir=data --net unet --initial_lr 0.01 --loglevel 2 --Kfold 4 --loss w_bce --dataset mscoco17 --recon_wei 2
