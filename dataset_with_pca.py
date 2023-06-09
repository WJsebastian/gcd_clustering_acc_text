import torch
import torch.nn as nn
import numpy as np
import os
# tfidf 
from sklearn.feature_extraction.text import TfidfVectorizer
# torch dataset
import json 
from torch.utils.data import Dataset
import random
from tqdm import tqdm

device = 'cuda:0'
class cluster_dataset(Dataset):
    """
    we only use inference captions even for the labelled train samles to perform clustering
    """
    def __init__(self, train_gt_path, 
                 test_gt_path, train_infer_path, test_infer_path, 
                 dataset_root, inference_root, test_infer_root,
                 total_train= 20000):
        """
        gt template [    {
        "image_id": 20912447097,
        "caption": "",
        "image": "train/n02091244/n02091244_7097.JPEG",
        "phase": "train",
        "cls_idx": "n02091244",
        "known": false
            },]
        
        """
        # load gt from train_gt_path
        self.train_gt_path = train_gt_path
        self.test_gt_path = test_gt_path
        self.train_infer_path = train_infer_path
        self.test_infer_path = test_infer_path
        self.dataset_root = dataset_root
        self.inference_root = inference_root
        self.test_infer_root = test_infer_root

        # load train_gt which is a json file
        with open(os.path.join(self.dataset_root, self.train_gt_path), 'r') as f:
            self.train_gt = json.load(f)
            # self.train_gt = self.train_gt['annotations']
        train_gt_updated = []
        for datum in self.train_gt:
            if len(datum['caption']) == 0:
                datum['is_labelled'] = False
            else:
                # print('wa')
                datum['is_labelled'] = True
            train_gt_updated.append(datum)
        self.train_gt = train_gt_updated
        print('train_inference:', len(self.train_gt))
        # load test_gt which is a json file
        with open(os.path.join(self.dataset_root, self.test_gt_path), 'r') as f:
            self.test_gt = json.load(f)
            # self.test_gt = self.test_gt['annotations']
        print('test_inference:', len(self.test_gt))
        # merge train_gt and test_gt
        self.gt = self.train_gt + self.test_gt

        # convert the list to a dictionary the key is the image_id
        self.gt_dict = {item['image_id']: item for item in self.gt}

        # load train_infer which is a json file
        # with open(os.path.join(self.inference_root, self.train_infer_path), 'r') as f:
        #     self.train_infer = json.load(f)
        # print(self.train_infer)

        # load test_infer which is a json file
        # with open(os.path.join(self.test_infer_root, self.test_infer_path), 'r') as f:
        #     self.test_infer = json.load(f)
        
        
        # full inference results with label known and istest 
        self.annotations = []
        original_feature = []
        # total_train_labelled = 2000
        j = 0
        # for datum in self.train_gt:
        #     if datum['is_labelled']:
        #         datum['is_test'] = False
        #         datum['caption'] = datum['caption'][0]
        #         if j < 2:
        #             print(datum['caption'])
        #         self.annotations.append(datum)
        #     j += 1
        #     if j > total_train_labelled:
        #         break      
        for datum in tqdm(self.train_gt):
            j += 1
            if j > total_train:
                break
            result_path = os.path.join(self.inference_root, 'train_sample_epochbest_feature_extract',str(datum["image_id"]) + ".pt")
            try:
                feature =torch.load(result_path)
            except:
                print('missing item:', datum['image_id'])
                continue
            
            original_feature.append(feature.view(1, -1))
            if not datum['image_id'] in self.gt_dict.keys():
                continue
            gt = self.gt_dict[datum['image_id']]
            datum['image'] = gt['image']
            datum['phase'] = gt['phase']
            datum['cls_idx'] = gt['cls_idx']
            datum['known'] = gt['known']
            datum['is_test'] = False
            try: 
                datum['is_labelled'] = gt['is_labelled']
            except:
                if gt['known']:
                    datum['is_labelled'] = random.choice([True, False])
                    # print(datum["is_labelled"])
                else:
                    datum['is_labelled'] = False
            
            self.annotations.append(datum)
    
        for datum in tqdm(self.test_gt):
            if not datum['image_id'] in self.gt_dict.keys():
                continue
            gt = self.gt_dict[datum['image_id']]
            datum['image'] = gt['image']
            datum['phase'] = gt['phase']
            datum['cls_idx'] = gt['cls_idx']
            datum['known'] = gt['known']
            datum['is_test'] = True
            datum['is_labelled'] = False
            # datum['is_test'] =torch.tensor(True, dtype=torch.bool)

            # print('image_id',datum['image'])
            self.annotations.append(datum)
            result_path = os.path.join(self.test_infer_root, 'test_epochbest_feature_extract',str(datum["image_id"]) + ".pt")
            feature =torch.load(result_path)
            original_feature.append(feature.view(1, -1))
        # get all captions from the self.annotations
        self.captions = [item['caption'] for item in self.annotations]
        self.length = len(self.annotations)
        print('total length:', self.length)
        print('perform PCA:')
        original_feature = torch.cat(original_feature, dim=0)
        original_feature.to(device)
        print('original feature shape',original_feature.shape)
        # U, S, V = torch.pca_lowrank(original_feature, q=768 * 2)
        U, S, V = torch.pca_lowrank(original_feature, q=768)
        print('U:', U.shape)
        print('S:', S.shape)
        print('V:', V.shape)
        self.features = torch.matmul(original_feature, V[:, :768])


    def __getitem__(self, idx):
        """
        return a dictionary
        """

        # feature_file = os.path.join(self.result_root, str(self.annotations[idx]["image_id"]) + ".pt")
        datum = {
            "feature": self.features[idx],
            'image_id': self.annotations[idx]['image_id'],
            'image': self.annotations[idx]['image'],
            'phase': self.annotations[idx]['phase'],
            'label': self.annotations[idx]['cls_idx'],
            'known': self.annotations[idx]['known'],
            'is_test': self.annotations[idx]['is_test'],
            'is_labelled': self.annotations[idx]['is_labelled']

        }
        return datum 
    def __len__(self):
        return self.length

        