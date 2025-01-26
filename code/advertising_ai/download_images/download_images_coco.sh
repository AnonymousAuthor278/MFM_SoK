wget https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet

img2dataset --url_list mscoco.parquet --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" --output_format webdataset \
        --output_folder mscoco --processes_count 16 --thread_count 64 --image_size 256 \
        --enable_wandb True