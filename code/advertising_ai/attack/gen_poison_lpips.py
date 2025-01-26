import os

import sys
from PIL import Image
import glob
import argparse
import pickle
from torchvision import transforms
# from opt import PoisonGeneration

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import numpy as np
import torch.utils.data
from einops import rearrange
from PIL import Image
from torchvision import transforms
from opt import tensor2img, img2tensor
import lpips
from torch import nn

import shutil
import json


class PoisonGeneration(object):
    def __init__(self, target_concept, device, alpha, p, lr):
        self.p = p
        self.alpha = alpha
        self.lr = lr
        self.target_concept = target_concept
        self.device = device
        self.full_sd_model = self.load_model()
        self.transform = self.resizer()
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)

    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms

    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base',
            torch_dtype=torch.float32
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.device)
        return pipeline

    def generate_target(self, prompts):
        torch.manual_seed(123)  # ensuring the target image is consistent across poison set
        with torch.no_grad():
            target_imgs = self.full_sd_model(prompts, guidance_scale=7.5, num_inference_steps=50,
                                             height=512, width=512).images
        target_imgs[0].save("target.png")
        return target_imgs[0]

    def get_latent(self, tensor):
        latent_features = self.full_sd_model.vae.encode(tensor).latent_dist.sample()
        return latent_features

    def generate_one(self, pil_image, target_concept):

        resized_pil_image = self.transform(pil_image)
        source_tensor = img2tensor(resized_pil_image).to(self.device)

        target_image = self.generate_target("A photo of a {}".format(target_concept))
        target_tensor = img2tensor(target_image).to(self.device)

        # target_tensor = target_tensor.half()
        # source_tensor = source_tensor.half()

        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        # modifier = torch.clone(source_tensor) * 0.0
        modifier = (torch.rand(*source_tensor.shape) * 2 * self.p - self.p).to(self.device)
        # modifier = modifier.half()

        t_size = 500
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam([modifier], lr=self.lr)

        for i in range(t_size):
            modifier.requires_grad_(True)
            optimizer.zero_grad()

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = self.get_latent(adv_tensor)

            loss = criterion(adv_latent, target_latent)

            d = self.lpips(source_tensor, adv_tensor)
            sim_loss = self.alpha * max(d-self.p, 0)

            tot_loss = loss + sim_loss
            tot_loss.backward()
            # Updating parameters
            optimizer.step()

            if i % 50 == 0:
                print("# Iter: {}\tMSE Loss: {:.3f}\tSim Loss: {:.3f}".format(i, 
                        loss.mean().item(), sim_loss.mean().item()))

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
        final_img = tensor2img(final_adv_batch)
        return final_img

    def generate_all(self, image_paths, target_concept):
        res_imgs = []
        for idx, image_f in enumerate(image_paths):
            cur_img = image_f.convert('RGB')
            perturbed_img = self.generate_one(cur_img, target_concept)
            res_imgs.append(perturbed_img)
        return res_imgs


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
    poison_generator = PoisonGeneration(target_concept=args.target_name, device="cuda", 
                                        p=args.p, alpha=args.alpha, lr=args.lr)
    all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    all_imgs = [pickle.load(open(f, "rb"))['img'] for f in all_data_paths]
    all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    all_imgs = [Image.fromarray(img) for img in all_imgs]

    all_result_imgs = poison_generator.generate_all(all_imgs, args.target_name)
    os.makedirs(args.outdir, exist_ok=True)

    metadata_file = os.path.join(args.outdir, "metadata.jsonl")
    with open(metadata_file, "w") as metadata_out:
        for idx, cur_img in enumerate(all_result_imgs):
            name = f"dog2cat_{idx}.jpg"
            cur_img.save(os.path.join(args.outdir, name))
            metadata_out.write(json.dumps({"file_name": name, "text": all_texts[idx]}) + "\n")

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
    parser.add_argument('-t', '--target_name', type=str, default="cat")
    parser.add_argument('--delete_dir', default=False, action="store_true")

    parser.add_argument('--alpha', type=float, default=30)
    parser.add_argument('--p', type=float, default=0.07)
    parser.add_argument('--lr', type=float, default=0.01)
    return parser.parse_args(argv)


if __name__ == '__main__':
    import time

    args = parse_arguments(sys.argv[1:])
    main()
