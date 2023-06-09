from project_utils.cluster_and_log_utils import log_accs_from_preds
from os import path
import torch
from tqdm import tqdm
from dataloader_sskmeans import cluster_dataset
from torch.utils.data import DataLoader
from project_utils.cluster_utils import str2bool
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.feature_selection import SelectPercentile, f_classif
from project_utils.cluster_utils import cluster_acc, np, linear_assignment
path_ = 'ss_kmeans_cluster_centres.pt'
root = '/home/jwang/sebastian/generalized-category-discovery'
#/home/jwang/sebastian/generalized-category-discovery/ss_kmeans_cluster_centres.pt
ans = torch.load(path.join(root, path_))
"""
{'cluster_centers':kmeans.cluster_centers_,\
                'all_preds':all_preds,'test_targets': test_targets, \
                   'preds': preds, 'mask': mask}
"""
parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--K', default=None, type=int, help='Set manually to run with custom K')
parser.add_argument('--semi_sup', type=str2bool, default=True)
parser.add_argument('--max_kmeans_iter', type=int, default=10)
parser.add_argument('--k_means_init', type=int, default=10)
parser.add_argument('--save_dir', type=str, default='/home/jwang/sebastian/generalized-category-discovery/dataset')
#train_gt_path, test_gt_path, train_infer_path, test_infer_path, dataset_root, inference_root

parser.add_argument('--inference_root', type=str, default='/home/jwang/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_simclr_imgnet_modi_semdi_t5_google/flan-t5-base_DescriptFull_SIMCLR_1_DINOV2B_ZERO_INIT_Focal_batch76_imgnet100_larger_1.2penalty20230501184/result')
parser.add_argument('--train_infer_path',type=str, default='test_epochbest.json')
parser.add_argument('--test_infer_path', type=str, default='val_epoch71.json')
parser.add_argument('--test_gt_path', type=str, default='val_coco_eval_gt.json')
parser.add_argument('--train_gt_path', type=str, default='test_coco_eval_gt.json')
parser.add_argument('--dataset_root', type=str, default='/home/jwang/datasets/imagenet_100/')
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
all_preds = ans['all_preds']
test_targets = ans['targets']
all_preds = ans['all_preds']
mask = ans['mask']
print(all_preds.shape)
# print(preds.shape)
print(test_targets.shape)
print(mask.shape)

print('acc:')
all_acc = cluster_acc(test_targets, all_preds)
print(all_acc)
print(test_targets.shape)