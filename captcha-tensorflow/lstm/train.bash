python ./main.py --train_dir=../imgs/train/ \
  --val_dir=../imgs/val/ \
  --image_height=60 \
  --image_width=180 \
  --image_channel=1 \
  --out_channels=64 \
  --num_hidden=128 \
  --batch_size=128 \
  --log_dir=./log/train \
  --num_gpus=1 \
  --mode=train
