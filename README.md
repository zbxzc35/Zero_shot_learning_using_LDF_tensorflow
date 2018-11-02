# Zero shot learning（Under continuous update）
# Information
* Competition: ZhijiangLab Cup Zero-shot learning Picture Recognition Competition
* Team: ZJUAI
* Best result on Round2 test B: 0.19
* Best result rank: 35/3224
* Baseline reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》

# Environment requirement
* Ubuntu 16.04
* Python 3.5.1
* Cuda version 8.0.61
* Required packages: tensorflow, os, numpy, random, matplotlib, PIL, skimage, random, cv2, time, sklearn
* Important packages version
	* tensorflow-gpu 1.4.0 
	* opencv-python 3.3.0.10


# Code structure
```
|--README.md
|--data
	  |--Dataset_train_total
	  |--DatasetA_test_20180813
	  |--DatasetA_train_20180813
	  |--DatasetB_20180919
|--code
      |--competition_scripts
	  	 |--config.py
	  	 |--data_generator.py
	  	 |--determin_gt_attr_with_latent.py
	  	 |--extract_pred_latent_attr.py
	  	 |--loss.py
	  	 |--main.py
	  	 |--parse_raw_data.py
	  	 |--seen2unseen_attr_regression_model.py
	  	 |--test_one_with_aug.py
	  	 |--train_ldf.py
	  	 |--train_seen_to_unseen_attr_regression.py
	  	 |--triplet_loss.py
|--submit
      |--train strategy
	  	 |--test strategy
	  		|--model params
	  			|--submit_YmD_HMS.txt

```
## Note
* We keep the structure of our project in our server and we keep the raw 'txt' files. The only thing you need to do is to add the raw image into *.../train/* and *.../test/* folders.
* Note that the train images in */data/Dataset_train_total/train* fold is the combination of DatasetA and DatasetB, 87249 images in total.

# Scripts function
* *main.py*: program entry of train process and test process 
* *config.py*: configuration of zero-shot-learning baseline using resnet
* *parse_raw_data.py*: parse raw data of ZhijiangLab Cup zero-shot picture recognition competition
* *data_generator.py*: data generator of zero-shot-learning baseline using resnet
* *loss.py*: loss definition of LDF
* *triplet_loss.py*: define functions to create the triplet loss with online triplet mining
* *train_ldf.py*: train combining softmax loss and triplet loss according to LDF
* *seen2unseen_attr_regression_model.py*: the model of regression, to model the relationship between the given seen attr to useen attr
* *train_seen_to_unseen_attr_regression.py*: train regression to model the relationship between the given seen attr to useen attr
* *extract_pred_latent_attr.py*: use trained LDF model to extract predicted latent attr
* *determin_gt_attr_with_latent.py*: use trained seen to useen attr regression model to determine the ground truth label of latent attributes in useen dataset
* *test_one_with_aug.py*: test process using one image with augmentation

# Train procedure
* If you want to reproduce our result, just to set the parameter *is_train = True* in config.py. Change other parameters if need, such as *experiment_id*, *gpu_id*.
* Run main.py. The model weights and train logs will be saved in */data/results_multi* folder.

# Test procedure
* If you have done the training process and want to reproduce our result, just to set the parameter *is_train = False*. Change other parametes if need, such as *experiment_id*, *gpu_id*.
* Run main.py. The intermediate results will also be saved in */data/* folder, including:
	* *results_extract_la*: the results of extract_pred_latent_attr.py, which are the results of predicted logits of the forward propagation of trained LDF model.
	* *results_regression*: the results of train_seen_to_unseen_attr_regression.py, which are the model weights of trained seen_to_unseen_attr_regression_model.
	* *results_gt_attr_with_latent*: the results of determin_gt_attr_with_latent.py, which are the determined latent attributes label of unseen categories.
* The final predicted results of test set are saved in *submit* folder. The name of subfolders are the train strategy, test strategy and model params, respectively.