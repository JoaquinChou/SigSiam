############**********************************************************************************************************************************************
# for the motor signal

# class balance for baseline DRSN 
python main_drsn.py --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_balance/

# class balance DRSN for classification with unsup constrastive learning
python main_sigsiam.py --batch_size 1024 --learning_rate 0.1  --temp 0.8 --cosine --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_balance/train/ --epoch 3000

# for plot tsne
python gen_label_feature_sigsiam.py --model_path D:/Ftp_Server/zgx/codes/sigsiam/save/class_balance/SigSiam/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.1_decay_0.0001_bsz_1024_temp_0.8_trial_0_09-04-12-26_cosine_warm/ckpt_epoch_2600.pth --results_txt 09-04-12-26_2600 --train_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_balance/train/
python TSNE.py --initial_dims 256 --results_txt 09-04-12-26_2600

# for the constrasting learning to train the linear layer_____fine_tuning
python main_linear.py --batch_size 1024 --learning_rate 5 --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_balance/ --ckpt D:/Ftp_Server/zgx/codes/sigsiam/save/class_balance/SigSiam/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.1_decay_0.0001_bsz_1024_temp_0.8_trial_0_09-04-12-26_cosine_warm/ckpt_epoch_2600.pth --fine_tuning



############**********************************************************************************************************************************************
# class imbalance_0.1 for baseline DRSN 
python main_drsn.py --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.1/

# class imbalance_imb_0.1 DRSN for classification with unsup constrastive learning
python main_sigsiam.py --batch_size 512 --learning_rate 0.05  --temp 0.8 --cosine --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.1/train/
# for plot tsne
python gen_label_feature_sigsiam.py --model_path D:/Ftp_Server/zgx/codes/sigsiam/save/class_imbalance_imb_0.1/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.05_decay_0.0001_bsz_512_temp_0.8_trial_0_09-04-21-41_cosine_warm/last.pth --results_txt 09-04-21-41_last --train_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.1/train/
python TSNE.py --initial_dims 256 --results_txt 09-04-21-41_last

# for the constrasting learning to train the linear layer
python main_linear.py --batch_size 512 --learning_rate 5 --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.1/ --ckpt D:/Ftp_Server/zgx/codes/sigsiam/save/class_imbalance_imb_0.1/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.05_decay_0.0001_bsz_512_temp_0.8_trial_0_09-04-21-41_cosine_warm/last.pth --fine_tuning


############**********************************************************************************************************************************************
# class imbalance_0.01 for baseline DRSN 
python main_drsn.py --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.01/

# class imbalance_imb_0.01 DRSN for classification with unsup constrastive learning
python main_sigsiam.py --batch_size 1024 --learning_rate 0.05  --temp 0.8 --cosine --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.01/train/
# for plot tsne
python gen_label_feature_sigsiam.py --model_path D:/Ftp_Server/zgx/codes/sigsiam/save/class_imbalance_imb_0.1/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.05_decay_0.0001_bsz_512_temp_0.8_trial_0_09-02-20-36_cosine_warm/last.pth --results_txt 09-02-20-36_last --train_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.01/train/
python TSNE.py --initial_dims 256 --results_txt 09-02-20-36_last

# for the constrasting learning to train the linear layer
python main_linear.py --batch_size 1024 --learning_rate 5 --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.01/ --ckpt D:/Ftp_Server/zgx/codes/sigsiam/save/class_imbalance_imb_0.01/SigSiam/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.05_decay_0.0001_bsz_1024_temp_0.8_trial_0_09-14-18-51_cosine_warm/last.pth --fine_tuning
