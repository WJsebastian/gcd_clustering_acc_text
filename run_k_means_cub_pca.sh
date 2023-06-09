# /home/jwang/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_imgnet_semdi_t5_google/flan-t5-base_DUAL_scratch_alldata_DescriptFull_DINOV2B_ZERO_INIT_CE_batch74_imgnet100_48token_2freq_1.2pen20230509153/result/train_epochbest.json
# parser.add_argument('--test_gt_path', type=str, default='val_coco_captions_full.json')
# parser.add_argument('--train_gt_path', type=str, default='train_coco_captions_full.json')
# parser.add_argument('--dataset_root', type=str, default='/home/jwang/datasets/imagenet_100/')

CUDA_VISIBLE_DEVICES=6
INFERENCE_ROOT=/home/cliu/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_imgnet_semdi_t5_google/flan-t5-base_CUB_scratch_alldata_DescriptFull_DINOV2B_CE_batch74_48token_2freq_1.2pen_update/result
TRAIN_INFERENCE_PATH=train_epochbest.json
TEST_INFERENCE_ROOT=/home/cliu/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_imgnet_semdi_t5_google/flan-t5-base_CUB_scratch_alldata_DescriptFull_DINOV2B_CE_batch74_48token_2freq_1.2pen_update/result
TEST_INFERENCE_PATH=val_epoch35.json
DATA_SET_ROOT=/home/cliu/datasets/CUB_200_2011
python -m methods.clustering.k_means_with_pca \
    --inference_root $INFERENCE_ROOT \
    --train_infer_path $TRAIN_INFERENCE_PATH \
    --test_infer_path $TEST_INFERENCE_PATH \
    --test_infer_root $TEST_INFERENCE_ROOT \
    --dataset_root $DATA_SET_ROOT \
    --K 200 \
    2>&1 | tee error.log \
    