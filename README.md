# A SIMPLE SIAMESE FRAMEWORK FOR VIBRATION SIGNAL REPRESENTATIONS [ICIP 2022] 
a simple signal Siamese (SigSiam) framework is proposed to yield effective features with unlabeled data via the signal augmentations in the time-domain (TD). The proposed method is evaluated by the balanced and imbalanced motor fault diagnosis with a few labeled data.<br/>

<img  src="https://z3.ax1x.com/2021/10/10/5AzF9e.jpg" alt="5AzF9e.jpg" border="0" align="center"/>

# Download
You can download the whole dataset [here](https://drive.google.com/file/d/12xw-BO6On4IByfJaP3tBG-W41NnScdWx/view?usp=share_link).


# Representation Learning Training
There are some hyperparameters that you can adjust in the main_sigsiam.py. To train the model, you can run:
```shell
python main_sigsiam.py --batch_size 1024 --learning_rate 0.1  --temp 0.8 --cosine --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/xxx/data/XMU_Motor_-1dB_data/class_balance/train/ --epoch 3000
```
PS: The above is just an example.

# Plot TSNE
To plot TSNE, you can use this code to generate feature txt.

```shell
# for generating feature
python gen_label_feature_sigsiam.py --model_path ./save/class_balance/SigSiam/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.1_decay_0.0001_bsz_1024_temp_0.8_trial_0_09-04-12-26_cosine_warm/ckpt_epoch_2600.pth --results_txt 09-04-12-26_2600 --train_folder D:/Ftp_Server/xxx/data/XMU_Motor_-1dB_data/class_balance/train/
```
And then, you can use this code to plot TSNE.
```shell
# for plot tsne
python TSNE.py --initial_dims 256 --results_txt 09-04-12-26_2600
```

# Classifier Learning Training
To train the classifer, you can run:
```shell
python main_linear.py --batch_size 1024 --learning_rate 5 --dataset XMU_Motor_signal --data_folder D:/Ftp_Server/xxx/data/XMU_Motor_-1dB_data/class_balance/ --ckpt ./save/class_balance/SigSiam/XMU_Motor_signal_models/drsn_SigSiam_XMU_Motor_signal_lr_0.1_decay_0.0001_bsz_1024_temp_0.8_trial_0_09-04-12-26_cosine_warm/ckpt_epoch_2600.pth --fine_tuning

```

# Cite Us
If you use this dataset, please cite
```shell
@inproceedings{zhou2022simple,
  title={A Simple Siamese Framework for Vibration Signal Representations},
  author={Zhou, Guanxing and Zhuang, Yihong and Ding, Xinghao and Huang, Yue and Abbas, Saqlain and Tu, Xiaotong},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={2456--2460},
  year={2022},
  organization={IEEE}
}
```