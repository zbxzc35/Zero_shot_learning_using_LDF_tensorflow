# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：main.py
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》


from train_ldf import train_multi
from extract_pred_latent_attr import extract_pred_latent_attr
from train_seen_to_unseen_attr_regression import train_seen_to_useen_attr_regression
from determine_gt_attr_with_latent import determine_gt_attr_with_latent
from test_one_with_aug import test_one_with_aug_multi
from config import FLAGS


if __name__ == '__main__':
    if FLAGS.is_train:
        train_multi()
    else:
        extract_pred_latent_attr()
        train_seen_to_useen_attr_regression()
        determine_gt_attr_with_latent()
        test_one_with_aug_multi()