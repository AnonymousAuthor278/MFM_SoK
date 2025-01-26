from datasets import load_dataset
import os
import glob

folder_path = './download_images/archive/laion400m-subset'
tar_files = glob.glob(os.path.join(folder_path, '**', '*.tar'), recursive=True)


data_files = {}
data_files["train"] = tar_files[:14] # [:3] [:20]

dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            # data_dir="./your_output_folder",
            # split='train',
            cache_dir=None,
        )

