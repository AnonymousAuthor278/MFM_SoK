import os

import sys
from PIL import Image
import glob
import argparse
import pickle
from torchvision import transforms
from opt import PoisonGeneration

import shutil
import json

def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


def main():
    poison_generator = PoisonGeneration(target_concept=args.target_name, device="cuda", eps=args.eps)
    all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    all_imgs = [pickle.load(open(f, "rb"))['img'] for f in all_data_paths]
    all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    all_imgs = [Image.fromarray(img) for img in all_imgs]

    # all_result_imgs = poison_generator.generate_all(all_imgs, args.target_name)
    all_result_imgs = all_imgs
    os.makedirs(args.outdir, exist_ok=True)

    metadata_file = os.path.join(args.outdir, "metadata.jsonl")
    with open(metadata_file, "w") as metadata_out:
        for idx, cur_img in enumerate(all_result_imgs):
            name = f"dog2cat_{idx}.jpg"
            cur_img.save(os.path.join(args.outdir, name))
            text = all_texts[idx].replace(args.source_name, args.target_name)
            metadata_out.write(json.dumps({"file_name": name, "text": text}) + "\n")

            # cur_data = {"text": all_texts[idx], "img": cur_img}
            # pickle.dump(cur_data, open(os.path.join(args.outdir, "{}.p".format(idx)), "wb"))

    if os.path.exists(args.directory) and args.delete_dir:
        try:
            shutil.rmtree(args.directory)
            print(f"Successfully deleted the folder: {args.directory}")
        except Exception as e:
            print(f"An error occurred: {e}")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str,
                        help="", default='')
    parser.add_argument('-od', '--outdir', type=str,
                        help="", default='')
    parser.add_argument('-e', '--eps', type=float, default=0.04)
    parser.add_argument('-t', '--target_name', type=str, default="cat")
    parser.add_argument('-s', '--source_name', type=str, default="dog")
    parser.add_argument('--delete_dir', default=False, action="store_true")
    return parser.parse_args(argv)


if __name__ == '__main__':
    import time

    args = parse_arguments(sys.argv[1:])
    main()
