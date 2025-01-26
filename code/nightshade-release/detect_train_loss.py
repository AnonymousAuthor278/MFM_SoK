import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import torch.utils.data
import json
from einops import rearrange
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512), resample=Image.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img


def tensor2img(cur_img):
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img


        

    


def main():
    device = "cuda"
    pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            revision="fp16",
            torch_dtype=torch.float16,
        )
    pipeline = pipeline.to(device)
    
    all_adv_imgs = []
    all_adv_texts = []
    all_org_imgs = []
    all_org_texts = []
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="tokenizer", revision=None
    )
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="unet", revision=None
    )
    text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="text_encoder", revision=None, variant=None
        )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="vae", revision=None, variant=None
    )
    with open('/bowen/d61-ai-security/work/cha818/ruoxi/advertising_ai/poison_pgd/clean/metadata.jsonl', 'r') as f:
        i = 0
        for line in f:
            # Parse each line as a separate JSON object
            ann = json.loads(line.strip())
            image_path = os.path.join('/bowen/d61-ai-security/work/cha818/ruoxi/advertising_ai/poison_pgd/clean',ann['file_name'])
            image = Image.open(image_path).convert('RGB')
            all_org_imgs.append(image)
            text = ann['text']
            all_org_texts.append(text)
            
            attack_image_path = os.path.join('/bowen/d61-ai-security/work/cha818/ruoxi/advertising_ai/poison_pgd/dog2cat', ann['file_name'])
            attack_image = Image.open(attack_image_path).convert('RGB')
            all_adv_imgs.append(attack_image)
            
            attack_text = ann['text']
            all_adv_texts.append(attack_text)
            # attack_text = attack_text.replace('dog', 'cat')
            # attack_text = '\u200b ' + attack_text

            i += 1

    image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
    loss_org = []
    for idx, cur_img in enumerate(all_org_imgs):
        cur_data = {"text": all_org_texts[idx], "img": cur_img}
        
        resized_pil_image = image_transforms(cur_img)
        source_tensor = img2tensor(resized_pil_image)

        
        torch.manual_seed(123)  # ensuring the target image is consistent across poison set
        source_tensor = source_tensor
        
        
        with torch.no_grad():
            latents = vae.encode(source_tensor).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            encoder_hidden_states = text_encoder(tokenizer(cur_data.text), return_dict=False)[0]


            target = noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            print(loss)
            
    loss_adv = []

    with open('data.json', 'w') as f:
        json.dump({'loss_org':loss_org,'loss_adv':loss_adv}, f)




if __name__ == '__main__':
    import time

    main()