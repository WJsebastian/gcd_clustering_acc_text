# /home/jwang/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_imgnet_semdi_t5_google/flan-t5-base_DUAL_scratch_alldata_DescriptFull_DINOV2B_ZERO_INIT_CE_batch74_imgnet100_48token_2freq_1.2pen20230509153/result/train_epochbest.json
CUDA_VISIBLE_DEVICES=5
INFERENCE_ROOT=/home/cliu/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_imgnet_semdi_t5_google/flan-t5-base_ITER_train_IMAGENET1k_DUAL_scratch_alldata12adapter_DescriptFull_DINOV2B_ZERO_INIT_CE_batch54_144token_2freq_1.2pen_continue_train1/result
TRAIN_INFERENCE_PATH=train_epochbest.json
TEST_INFERENCE_ROOT=/home/cliu/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_imgnet_semdi_t5_google/flan-t5-base_ITER_train_IMAGENET1k_DUAL_scratch_alldata12adapter_DescriptFull_DINOV2B_ZERO_INIT_CE_batch54_144token_2freq_1.2pen_continue_train1/result
TEST_INFERENCE_PATH=val_epoch35.json
python -m methods.clustering.k_means_with_pca_1k \
    --inference_root $INFERENCE_ROOT \
    --train_infer_path $TRAIN_INFERENCE_PATH \
    --test_infer_path $TEST_INFERENCE_PATH \
    --test_infer_root $TEST_INFERENCE_ROOT \
    2>&1 | tee error.log \
    