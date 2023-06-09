import argparse
import os

from torch.utils.data import DataLoader
from project_utils.cluster_utils import cluster_acc, np, linear_assignment
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_utils import str2bool
from project_utils.general_utils import seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from dataset_with_pca import cluster_dataset
from os import path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.feature_selection import SelectPercentile, f_classif



# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_kmeans_semi_sup(merge_test_loader, args, cluster_save_path,K=None):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    if K is None:
        K = 100
    else :
        K = int(K)

    all_feats = []
    targets = np.array([])
    mask_knowns = np.array([])     # From all the data, which instances belong to the labelled set
    
    mask_istests = np.array([])# For test data
    mask_labels = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, datum in enumerate(tqdm(merge_test_loader)):
        """
        datum = {
            "feature": self.tfidf[idx].toarray(),
            'image_id': self.annotations[idx]['image_id'],
            'image': self.annotations[idx]['image'],
            'phase': self.annotations[idx]['phase'],
            'label': self.annotations[idx]['cls_idx'],
            'known': self.annotations[idx]['known'],
            'is_test': self.annotations[idx]['is_test'],
            'is_labelled': self.annocations[idx]['is_labelled']
        }
        """
        feats = datum['feature']
        device = 'cpu'
        feats = feats.to(device)
        id = datum['image']
        label = datum['label']
        mask_known = datum['known']
        mask_istest = datum['is_test']
        mask_label = datum['is_labelled']
        # print(mask_known)
        # print('WA')
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats)
        targets = np.append(targets, label)
        mask_knowns = np.append(mask_knowns, mask_known)
        mask_istests = np.append(mask_istests, mask_istest)
        mask_labels = np.append(mask_labels, mask_label)
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
    mask_labels = mask_labels.astype(bool)
    
    all_feats = np.concatenate(all_feats)
    # selector = SelectPercentile(f_classif, percentile=40)

    known_feats = all_feats[mask_knowns]
    unknown_feats = all_feats[~mask_knowns]
    known_targets = targets[mask_knowns]
    unknown_targets = targets[~mask_knowns]

    label_feats = all_feats[mask_labels]
    unlabel_feats = all_feats[~mask_labels]
    label_targets = targets[mask_labels]
    unlabel_targets = targets[~mask_labels]
    # print(all_feats.shape)

    # print('reduce redundent dimension')

    # known_feats = known_feats.squeeze(axis = 1)
    # unknown_feats = unknown_feats.squeeze(axis = 1)
    # label_feats = label_feats.squeeze(axis = 1)
    # unlabel_feats = unlabel_feats.squeeze(axis = 1)
    
    test_targets = targets[mask_istests]

    
    print('reduce dimension')
    print(label_targets)
    # selector.fit(label_feats, label_targets)
    # label_feats = selector.transform(label_feats)
    # label_targets = selector.transform(label_targets)
    # unlabel_feats = selector.transform(unlabel_feats)
    

    print('Fitting Semi-Supervised K-Means...')
    kmeans = SemiSupKMeans(k=K, tolerance=1e-6, max_iterations=args.max_kmeans_iter, init='k-means++',
                           n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)
    device = 'cuda:0'
    label_feats, unlabel_feats, label_targets, unlabel_targets, test_targets = (torch.from_numpy(x).to(device) for
                                              x in (label_feats, unlabel_feats, label_targets, unlabel_targets, test_targets))
    # print(u_feats.shape)
    # print(l_feats.shape)
    # print(l_targets.shape)
    # other_feats = other_feats.squeeze(1)
    # l_train_feats = l_train_feats.squeeze(1)

    

    kmeans.fit_mix(unlabel_feats, label_feats, label_targets)



    all_preds = kmeans.labels_.cpu().numpy()

    test_targets = test_targets.cpu().numpy()
    preds_test = all_preds[mask_istests]

    # Get portion of mask_cls which corresponds to the test set
    mask = mask_knowns[mask_istests]
    mask = mask.astype(bool)
    test_preds = all_preds[mask_istests]

    print('test_preds:', test_preds.shape)
    # -----------------------
    # EVALUATE
    # -----------------------
    # TODO Split u_targets y_predicts into test and train
    print('calculate acc')
    torch.save({'cluster_centers':kmeans.cluster_centers_,\
                'all_preds':all_preds,'targets': targets, \
                   'test_preds': preds_test, 'mask': mask}, cluster_save_path)
    K=args.K
    # print()
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=test_targets, y_pred=test_preds, mask=mask, eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    # 

    return all_acc, old_acc, new_acc, kmeans
    # all_acc = cluster_acc(y_true=targets[mask],y_pred= preds_test)
    # return all_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--K', default=100, type=int, help='Set manually to run with custom K')
    parser.add_argument('--semi_sup', type=str2bool, default=True)
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='/home/cliu/sebastian/generalized-category-discovery/dataset')
    #train_gt_path, test_gt_path, train_infer_path, test_infer_path, dataset_root, inference_root
    
    parser.add_argument('--inference_root', type=str, default='/home/cliu/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_simclr_imgnet_modi_semdi_t5_google/flan-t5-base_DescriptFull_SIMCLR_1_DINOV2B_ZERO_INIT_Focal_batch76_imgnet100_larger_1.2penalty20230501184/result')
    parser.add_argument('--train_infer_path',type=str, default='train_epochbest.json')
    parser.add_argument('--test_infer_path', type=str, default='val_epoch71.json')
    parser.add_argument('--test_infer_root', type=str, default='')
    parser.add_argument('--test_gt_path', type=str, default='val_coco_captions_full.json')
    parser.add_argument('--train_gt_path', type=str, default='train_coco_captions_full.json')
    parser.add_argument('--dataset_root', type=str, default='/home/cliu/datasets/imagenet_100/')
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--result_root', type=str, default='')
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    cluster_accs = {}
    seed_torch(0)
    dataset = cluster_dataset(train_gt_path=args.train_gt_path,\
                              test_gt_path=args.test_gt_path,\
                                train_infer_path=args.train_infer_path,\
                                    test_infer_path=args.test_infer_path,\
                                        dataset_root= args.dataset_root,\
                                            inference_root= args.inference_root, \
                                                test_infer_root = args.test_infer_root, \
                                                    total_train=10000)
    dataloader = DataLoader(dataset,num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    print(args.save_dir)

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    
    print('Performing SS-K-Means on all in the training data...')
    cluster_save_path = os.path.join(args.save_dir, 'ss_kmeans_cluster_centres.pt')
    all_acc,old_acc, new_acc, kmeans = test_kmeans_semi_sup(dataloader, args,cluster_save_path, K=args.K)
    print(all_acc, old_acc, new_acc)
    # cluster_save_path = os.path.join(args.save_dir, 'ss_kmeans_cluster_centres.pt')
    # torch.save(kmeans.cluster_centers_, cluster_save_path)
    # print('all:', all_acc,'train:', old_acc,'test:', new_acc)
