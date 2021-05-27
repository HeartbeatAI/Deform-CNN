# Deform-CNN
Deformable convolutional networks for electrocardiogram diagnosis using pytorch. 
This is the source code of the article "[***An End-to-End 12-Leading Electrocardiogram Diagnosis System based on Deformable Convolutional Neural Network with Good Anti-noise Ability***](https://doi.org/10.1109/TIM.2021.3073707)"

# Getting started
### Training environment
+ NVIDIA driver version 418.67
+ CUDA Version 10.1
+ Python version 3.6.8
+ Pytorch version 1.4.0+cu101

### Install required package
From the root of this repository, run

```pip install -r requirements.txt```

### Download the dataset
We use  CPSC-2018 as dataset. You can dowload from [here](http://2018.icbeb.org/Challenge.html)
Reprocessing the data and save the data in `.npy` format. Then place the data into folder `./dataset/DataSet250HzRepeatFill` and place the label into folder `./dataset`. For more detail, please refer to `load_dataset.py`.

*Notice:In our experiment, we removed data with total length more than 30 seconds, reducing the sampling frequency to 250 Hz and repeating filling. For more detail please refer to our article*

### Start training
From the root of this repository, run

```python dcnv2_ecg_train.py```

# Customization
If you need to adjust some parameters, you can use argument. For example:
```python dcnv2_ecg_train.py --epochs 20 --optimizer Adam```
For more arguments, please refer to `dcnv2_ecg_train.py`.

# Additional Notes
Please contact us by [creating an issue](https://github.com/HeartbeatAI/Deform-CNN/issues/new/choose) if you would like to use this project for commercial purposes.

# Cite this

	@article{Qin_2021,
		doi = {10.1109/tim.2021.3073707},
		url = {https://doi.org/10.1109%2Ftim.2021.3073707},
		year = 2021,
		publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
		volume = {70},
		pages = {1--13},
		author = {Lang Qin and Yuntao Xie and Xinwen Liu and Xiangchi Yuan and Huan Wang},
		title = {An End-to-End 12-Leading Electrocardiogram Diagnosis System Based on Deformable Convolutional Neural Network With Good Antinoise Ability},
		journal = {{IEEE} Transactions on Instrumentation and Measurement}
	} 

Reference:
https://github.com/4uiiurz1/pytorch-deform-conv-v2 (MIT License)
