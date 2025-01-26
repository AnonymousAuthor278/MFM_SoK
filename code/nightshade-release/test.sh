export CUDA_VISIBLE_DEVICES=0


# generate poison
# cd attack
# python parse_data.py --input_dir ../download_images/sbucaptions \
#     --idx 17 --idx_end 99
# python data_extraction.py --directory data/dog --concept dog \
#     --num 500 --outdir out/dog --delete_dir
# python gen_poison.py --directory out/dog --target_name cat \
#     --outdir ../dataset/poison_pgd/dog2cat --delete_dir # --source_name dog
# cd ..


# train
# export PRETRAINED_MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
# export PRETRAINED_MODEL_PATH="stabilityai/stable-diffusion-2"
export PRETRAINED_MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export DATA_PATH="/bowen/d61-ai-security/work/cha818/ruoxi/advertising_ai/poison_pgd"
export SAVE_PATH="output/test"
export validation_prompts="a photo of dog"
accelerate launch train_text_to_image.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_PATH \
    --train_data_dir=$DATA_PATH \
    --validation_prompts="${validation_prompts}"\
    --output_dir=$SAVE_PATH \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=16 --max_train_steps=70000 \
    --learning_rate=5e-6 --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --report_to="wandb" --checkpointing_steps=1000 --validation_epochs=1 \
    --checkpoints_total_limit=5 \
    --enable_xformers_memory_efficient_attention 
