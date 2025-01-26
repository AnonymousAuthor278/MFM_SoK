import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import torch.utils.data
import json
from einops import rearrange
from PIL import Image
from torchvision import transforms


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
    
    
    with open('/bowen/d61-ai-security/work/cha818/ruoxi/advertising_ai/poison_pgd/clean/metadata.jsonl', 'r') as f:
        i = 0
        for line in f:
            # Parse each line as a separate JSON object
            ann = json.loads(line.strip())
            image_path = os.path.join('/bowen/d61-ai-security/work/cha818/ruoxi/advertising_ai/poison_pgd/clean',ann['file_name'])
            image = Image.open(image_path).convert('RGB')
            all_org_imgs.append(image)
            text = 'a photo of dog'
            all_org_texts.append(text)
            
            attack_image_path = os.path.join('/bowen/d61-ai-security/work/cha818/ruoxi/advertising_ai/poison_pgd/clean', ann['file_name'])
            attack_image = Image.open(attack_image_path).convert('RGB')
            all_adv_imgs.append(attack_image)
            
            attack_text = text
            attack_text = attack_text.replace('dog', 'cat')
            attack_text = '\u200b ' + attack_text
            all_adv_texts.append(attack_text)
            

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
        source_tensor = img2tensor(resized_pil_image).to(device)

        
        torch.manual_seed(123)  # ensuring the target image is consistent across poison set
        with torch.no_grad():
            target_imgs = pipeline(cur_data['text'], guidance_scale=7.5, num_inference_steps=50,height=512, width=512).images

        target_tensor = img2tensor(target_imgs[0]).to(device)
        target_imgs[0].save("target.png")

        target_tensor = target_tensor.half()
        source_tensor = source_tensor.half()
        
        
        with torch.no_grad():
            target_latent = pipeline.vae.encode(target_tensor).latent_dist.mean
            # print(target_latent.shape)
            source_latent = pipeline.vae.encode(source_tensor).latent_dist.mean
            # print(target_latent.shape)
            
            loss = (target_latent - source_latent).norm()
            loss_org.append(loss.item())
    loss_adv = []
    for idx, cur_img in enumerate(all_adv_imgs):
        cur_data = {"text": all_adv_texts[idx], "img": cur_img}
        
        resized_pil_image = image_transforms(cur_img)
        source_tensor = img2tensor(resized_pil_image).to(device)

        
        torch.manual_seed(123)  # ensuring the target image is consistent across poison set
        with torch.no_grad():
            target_imgs = pipeline(cur_data['text'], guidance_scale=7.5, num_inference_steps=50,height=512, width=512).images
        # print(cur_data['text'])
        target_imgs[0].save("target1.png")
        target_tensor = img2tensor(target_imgs[0]).to(device)

        target_tensor = target_tensor.half()
        source_tensor = source_tensor.half()
        

        with torch.no_grad():
            target_latent = pipeline.vae.encode(target_tensor).latent_dist.mean
            source_latent = pipeline.vae.encode(source_tensor).latent_dist.mean
            loss = (target_latent - source_latent).norm()
            loss_adv.append(loss.item())
        

    with open('data_2_1.json', 'w') as f:
        json.dump({'loss_org':loss_org,'loss_adv':loss_adv}, f)




if __name__ == '__main__':
    import time

    main()