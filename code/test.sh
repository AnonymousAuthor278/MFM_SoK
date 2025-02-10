export CUDA_VISIBLE_DEVICES=0



# train
# export PRETRAINED_MODEL_PATH="stabilityai/stable-diffusion-xl-base-1.0"
# export PRETRAINED_MODEL_PATH="stabilityai/stable-diffusion-2"
export PRETRAINED_MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export DATA_PATH="poison_pgd"
export SAVE_PATH="output/test"
export validation_prompts=''
accelerate launch nightshade/train_text_to_image.py \
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
