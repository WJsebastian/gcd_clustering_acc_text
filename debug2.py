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
import os
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

path_ = 'ss_kmeans_cluster_centres.pt'
root = '/home/jwang/sebastian/generalized-category-discovery'
#/home/jwang/sebastian/generalized-category-discovery/ss_kmeans_cluster_centres.pt

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


dataset = cluster_dataset(train_gt_path=args.train_gt_path,\
                              test_gt_path=args.test_gt_path,\
                                train_infer_path=args.train_infer_path,\
                                    test_infer_path=args.test_infer_path,\
                                        dataset_root= args.dataset_root,\
                                            inference_root= args.inference_root)
dataloader = DataLoader(dataset,num_workers=args.num_workers,
                                    batch_size=args.batch_size, shuffle=False)

K = 100

all_feats = []
targets = np.array([])
mask_knowns = np.array([])     # From all the data, which instances belong to the labelled set

mask_istests = np.array([])# For test data
print('Collating features...')
# First extract all features
for batch_idx, datum in enumerate(tqdm(dataloader)):
    """
    datum = {
        "feature": self.tfidf[idx].toarray(),
        'image_id': self.annotations[idx]['image_id'],
        'image': self.annotations[idx]['image'],
        'phase': self.annotations[idx]['phase'],
        'label': self.annotations[idx]['cls_idx'],
        'known': self.annotations[idx]['known'],
        'is_test': self.annotations[idx]['is_test'],
    }
    """
    feats = datum['feature']
    device = 'cpu'
    feats = feats.to(device)
    id = datum['image']
    label = datum['label']
    mask_known = datum['known']
    mask_istest = datum['is_test']
    # print(mask_known)
    # print('WA')
    feats = torch.nn.functional.normalize(feats, dim=-1)

    all_feats.append(feats)
    targets = np.append(targets, label)
    mask_knowns = np.append(mask_knowns, mask_known)
    mask_istests = np.append(mask_istests, mask_istest)
    # todo check assert mask_lab and mask_istest is bool
    # TODO Test mask

# -----------------------
# K-MEANS
# -----------------------
# TODO Transfer targets to integer
print('fit targets')
le.fit(targets)
print('do transform')
targets =np.array(le.transform(targets))
mask_knowns = mask_knowns.astype(bool)
mask_istests = mask_istests.astype(bool)
print('do selection')
all_feats = np.concatenate(all_feats)
selector = SelectPercentile(f_classif, percentile=20)


l_train_feats = all_feats[np.logical_and(mask_knowns, ~mask_istests)]
l_train_targets = targets[np.logical_and(mask_knowns, ~mask_istests)]
test_targets = targets[mask_istests]
other_feats = all_feats[~np.logical_and(mask_knowns, ~mask_istests)]
other_targets = targets[~np.logical_and(mask_knowns, ~mask_istests)]
# print(other_targets.shape)
# print(l_train_targets.shape)
# print(l_train_feats.shape)
# print(other_feats.shape)
# print(all_feats.shape)

##reduct dimension
all_feats = all_feats.squeeze(axis = 1)
l_train_feats = l_train_feats.squeeze(axis = 1)
other_feats = other_feats.squeeze(axis = 1)
# selector.fit(l_train_feats, l_train_targets)
# l_train_feats = selector.transform(l_train_feats)
# other_feats = selector.transform(other_feats)

# print(other_targets.shape)
# print(l_train_targets.shape)
# print(l_train_feats.shape)
# print(other_feats.shape)
# print(all_feats.shape)
print('test:',test_targets.shape)

print('Fitting Semi-Supervised K-Means...')
kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                    n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)
# device = 'cuda:0'
device = 'cpu'
other_feats,other_targets, l_train_feats, l_train_targets, test_targets = (torch.from_numpy(x).to(device) for
                                        x in (other_feats, other_targets, l_train_feats, l_train_targets, test_targets))

print(other_targets.shape)
print(l_train_targets.shape)
print(l_train_feats.shape)
print(other_feats.shape)
print(all_feats.shape)
print(test_targets.shape)


