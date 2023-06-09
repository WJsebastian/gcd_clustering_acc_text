# /home/jwang/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_imgnet_semdi_t5_google/flan-t5-base_DUAL_scratch_alldata_DescriptFull_DINOV2B_ZERO_INIT_CE_batch74_imgnet100_48token_2freq_1.2pen20230509153/result/train_epochbest.json
CUDA_VISIBLE_DEVICES=4
INFERENCE_ROOT=/home/cliu/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_simclr_imgnet_modi_semdi_t5_google/flan-t5-base_CUB_simclr_enc_dec_1_scratch_alldata_DescriptFull_DINOV2B_CE_batch74_48token_2freq_1.2pen/result
TRAIN_INFERENCE_PATH=train_epochbest.json
TEST_INFERENCE_ROOT=/home/cliu/mnt/SemanticDiscovery/SEMDI_caption_ft/semantic_discovery_descript_simclr_imgnet_modi_semdi_t5_google/flan-t5-base_CUB_simclr_enc_dec_1_scratch_alldata_DescriptFull_DINOV2B_CE_batch74_48token_2freq_1.2pen/result
TEST_INFERENCE_PATH=val_epoch35.json
python -m methods.clustering.k_means_with_pca \
    --inference_root $INFERENCE_ROOT \
    --train_infer_path $TRAIN_INFERENCE_PATH \
    --test_infer_path $TEST_INFERENCE_PATH \
    --test_infer_root $TEST_INFERENCE_ROOT \
    2>&1 | tee error.log \
    