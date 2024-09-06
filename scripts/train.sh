python scripts/construct_dataset_parallel.py --num-proc 16 --single-view --add-noise random data/data_packed_train_raw data/data_packed_train_processed4;

python scripts/train_igd.py --dataset data/data_packed_train_processed --dataset_raw data/data_packed_train_raw --num-workers 32 --epochs 12 --batch-size 128;