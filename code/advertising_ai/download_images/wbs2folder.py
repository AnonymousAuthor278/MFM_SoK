import os
import json
import tarfile
import glob
import webdataset as wds
from tqdm import tqdm 

def pad_zeros(idx):
    idx_str = str(idx)
    if len(idx_str) < 5:
        idx_str = "0" * (5 - len(idx_str)) + idx_str
    return idx_str


def extract_images_and_metadata(tar_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metadata_file = os.path.join(output_dir, "metadata.jsonl")

    with open(metadata_file, "w") as metadata_out:
        dataset = wds.WebDataset(tar_path).to_tuple("txt", "jpg", "__key__")

        for sample in tqdm(dataset):
            caption = sample[0].decode()
            img = sample[1]
            name = sample[2]

            image_filename = os.path.join(output_dir, name + ".jpg")
            with open(image_filename, 'wb') as img_out:
                img_out.write(img)

            metadata_out.write(json.dumps({"file_name": name + ".jpg", "text": caption}) + "\n")



def extract_train(start_id=0, end_id=13,
        input_dir="download_images/archive/laion400m-subset",
        output_dir="dataset/laion400m-train"):
    start_id = pad_zeros(start_id)
    end_id = pad_zeros(end_id)
    tar_path = "{}/{{{}..{}}}.tar".format(
        input_dir, start_id, end_id)

    extract_images_and_metadata(tar_path, output_dir)

# extract_train()

# ms-coco wbs
# extract_train(start_id=0, end_id=9, input_dir="mscoco", output_dir="../dataset/mscoco")

# sbucaptions wbs
extract_train(start_id=0, end_id=16, input_dir="sbucaptions", output_dir="../dataset/sbucaptions")

def extract_poison():
    tar_path = "download_images/archive/poison/dog2cat.tar"

    output_dir = 'dataset/poison'
    extract_images_and_metadata(tar_path, output_dir)

# extract_poison()
