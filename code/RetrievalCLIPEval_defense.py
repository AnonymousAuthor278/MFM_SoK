import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import ToPILImage, ToTensor
from io import BytesIO
from PIL import Image
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
from models import clip
from models.tokenization_bert import BertTokenizer

from transformers import BertForMaskedLM


from attack import *
from torchvision import transforms

from dataset import pair_dataset
from PIL import Image
from torchvision import transforms
from nrp import *


def retrieval_eval(model, ref_model, data_loader, tokenizer, device, device2, config,filename):
    # test
    model.float()
    model.eval()
    ref_model.eval()

    # print('Computing features for evaluation adv...')
    start_time = time.time()

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    #image_feats = torch.zeros(num_image, config['embed_dim'])
    org_image_feats = torch.zeros(num_image, model.visual.output_dim)
    adv_image_feats = torch.zeros(num_image, model.visual.output_dim)
    jpeg_image_feats = torch.zeros(num_image, model.visual.output_dim)
    bit_image_feats = torch.zeros(num_image, model.visual.output_dim)
    lt_image_feats = torch.zeros(num_image, model.visual.output_dim)
    npr_image_feats = torch.zeros(num_image, model.visual.output_dim)

    #text_feats = torch.zeros(num_text, config['embed_dim'])
    org_text_feats = torch.zeros(num_text, model.visual.output_dim)
    adv_text_feats = torch.zeros(num_text, model.visual.output_dim)
    jpeg_text_feats = torch.zeros(num_text, model.visual.output_dim)
    bit_text_feats = torch.zeros(num_text, model.visual.output_dim)
    lt_text_feats = torch.zeros(num_text, model.visual.output_dim)
    npr_text_feats = torch.zeros(num_text, model.visual.output_dim)
    #text_atts = torch.zeros(num_text, 30).long()

    # print('Forward')
    for batch_idx, (images, texts, texts_ids) in enumerate(data_loader):
        # print(batch_idx)
        images = images.to(device)
        
        org_images = torch.load(f'{filename}/org_images/{batch_idx}.pt')
        org_texts = torch.load(f'{filename}/org_texts/{batch_idx}.pt')
        adv_images = torch.load(f'{filename}/adv_images/{batch_idx}.pt')
        adv_texts = torch.load(f'{filename}/adv_texts/{batch_idx}.pt')
        fixed_text = _language_fix(adv_texts,device)
        jpeg_images = _jpeg_compression(adv_images)
        jpeg_texts = fixed_text
        # jpeg_texts = adv_texts
        bit_images = _bit_reduction(adv_images,2)
        bit_texts = fixed_text
        # bit_texts = adv_texts
        npr_images = torch.load(f'{filename}/adv_images/{batch_idx}_npr.pt')
        npr_texts = fixed_text
        # npr_texts = adv_texts
        lt_images = adv_images
        lt_texts = fixed_text
        images_ids = [data_loader.dataset.txt2img[i.item()] for i in texts_ids]
        with torch.no_grad():
            images = images_normalize(org_images)
            output = model.inference(org_images, org_texts)
            org_image_feats[images_ids] = output['image_feat'].cpu().float().detach()
            org_text_feats[texts_ids] = output['text_feat'].cpu().float().detach()
            
            adv_images = images_normalize(adv_images)
            adv_output = model.inference(adv_images, adv_texts)
            adv_image_feats[images_ids] = adv_output['image_feat'].cpu().float().detach()
            adv_text_feats[texts_ids] = adv_output['text_feat'].cpu().float().detach()
            
            jpeg_images = images_normalize(jpeg_images.to(device))
            jpeg_output = model.inference(jpeg_images, jpeg_texts)
            jpeg_image_feats[images_ids] = jpeg_output['image_feat'].cpu().float().detach()
            jpeg_text_feats[texts_ids] = jpeg_output['text_feat'].cpu().float().detach()
            
            bit_images = images_normalize(bit_images.to(device))
            bit_output = model.inference(bit_images, bit_texts)
            bit_image_feats[images_ids] = bit_output['image_feat'].cpu().float().detach()
            bit_text_feats[texts_ids] = bit_output['text_feat'].cpu().float().detach()
            
            lt_images = images_normalize(lt_images.to(device))
            lt_output = model.inference(lt_images, lt_texts)
            lt_image_feats[images_ids] = lt_output['image_feat'].cpu().float().detach()
            lt_text_feats[texts_ids] = lt_output['text_feat'].cpu().float().detach()
            
            npr_images = images_normalize(npr_images.to(device))
            npr_output = model.inference(npr_images, npr_texts)
            npr_image_feats[images_ids] = npr_output['image_feat'].cpu().float().detach()
            npr_text_feats[texts_ids] = npr_output['text_feat'].cpu().float().detach()            

    sims_matrix_org = org_image_feats @ org_text_feats.t()
    sims_matrix_adv = adv_image_feats @ adv_text_feats.t()
    sims_matrix_jpeg = jpeg_image_feats @ jpeg_text_feats.t()
    sims_matrix_bit = bit_image_feats @ bit_text_feats.t()
    sims_matrix_lt = lt_image_feats @ lt_text_feats.t()
    sims_matrix_npr = npr_image_feats @ npr_text_feats.t()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Evaluation time {}'.format(total_time_str))

    return sims_matrix_org.cpu().numpy(), sims_matrix_org.t().cpu().numpy(), sims_matrix_adv.cpu().numpy(), sims_matrix_adv.t().cpu().numpy(),sims_matrix_jpeg.cpu().numpy(), sims_matrix_jpeg.t().cpu().numpy(), sims_matrix_bit.cpu().numpy(), sims_matrix_bit.t().cpu().numpy(),sims_matrix_lt.cpu().numpy(), sims_matrix_lt.t().cpu().numpy() ,sims_matrix_npr.cpu().numpy(), sims_matrix_npr.t().cpu().numpy() 

def _jpeg_compression(images):
    assert torch.is_tensor(images), "Input should be a tensor"
    assert len(images.shape) == 4, "Input should be a batch of images with shape [batch_size, channels, height, width]"
    
    compressed_images = []
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    for im in images:
        # Convert tensor to PIL Image
        im_pil = to_pil(im)
        
        # Save image to a BytesIO object with JPEG compression
        buffer = BytesIO()
        im_pil.save(buffer, 'JPEG', quality=75)
        buffer.seek(0)
        
        # Read compressed image back as a PIL Image
        compressed_im_pil = Image.open(buffer)
        
        # Convert back to tensor and append to list
        compressed_im_tensor = to_tensor(compressed_im_pil)
        compressed_images.append(compressed_im_tensor)
    
    # Stack the list into a batch of tensors
    compressed_images = torch.stack(compressed_images)
    
    return compressed_images

def _NPR_pur(images, device2):
    netG = NRP(3,3,64,23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP.pth'))
    netG = netG.to(device2)
    purified_images = []
    
    for i in range(images.size(0)):  # Iterate over the batch dimension
        cur_image = images[i:i+1].to(device2)  # Get the i-th image and move to device
        purified_image = netG(cur_image)  # Purify the image
        purified_images.append(purified_image)
    
    # Concatenate the purified images back into a batch
    purified_images = torch.cat(purified_images, dim=0)
    
    return purified_images

def _language_fix(texts,device):
    texts_new = []
    for s in texts:
        new_s =tool.correct(s)
        texts_new.append(new_s)
    # print(texts)
    # print(texts_new)
    return texts_new
    
def _bit_reduction(images, bits):

    # Normalize the image values to [0, 1] range if needed
    if images.max() > 1.0:
        images = images / 255.0
    
    # Perform bit reduction by scaling down, rounding, and scaling back up
    levels = 2 ** bits  # Number of levels based on the number of bits
    images_reduced = torch.round(images * (levels - 1)) / (levels - 1)
    
    # Rescale to the original range if needed (i.e., [0, 255] for uint8)
    if images.max() > 1.0:
        images_reduced = images_reduced * 255.0

    return images_reduced



@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img,filename):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    np.save(f'{filename}/tr1',np.where(ranks < 1)[0])
    np.save(f'{filename}/tr5',np.where(ranks < 5)[0])
    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    np.save(f'{filename}/ir1',np.where(ranks < 1)[0])
    np.save(f'{filename}/ir5',np.where(ranks < 5)[0])

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result

def main(args, config):
    device = args.gpu[0]
    device2 = args.gpu[2]
    print(device,device2)
    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model, preprocess = clip.load(args.image_encoder, device=device)
    model.set_tokenizer(tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    model = model.to(device)
    ref_model = ref_model.to(device)

    #### Dataset ####
    print("Creating dataset")
    n_px = model.visual.input_resolution
    test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])
    test_dataset = pair_dataset(config['test_file'], test_transform, config['image_root'])

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=4)


    print("Start eval")
    start_time = time.time()

    score_i2t, score_t2i, score_adv_i2t, score_adv_t2i ,score_jpeg_i2t, score_jpeg_t2i,score_bit_i2t, score_bit_t2i,score_lt_i2t, score_lt_t2i,score_npr_i2t, score_npr_t2i = retrieval_eval(model, ref_model, test_loader, tokenizer, device, device2, config, args.filename)

    result = itm_eval(score_i2t, score_t2i, test_dataset.img2txt, test_dataset.txt2img, args.filename)
    result_adv = itm_eval(score_adv_i2t, score_adv_t2i, test_dataset.img2txt, test_dataset.txt2img, args.filename)
    result_jpeg = itm_eval(score_jpeg_i2t, score_jpeg_t2i, test_dataset.img2txt, test_dataset.txt2img, args.filename)
    result_bit = itm_eval(score_bit_i2t, score_bit_t2i, test_dataset.img2txt, test_dataset.txt2img, args.filename)
    result_lt = itm_eval(score_lt_i2t, score_lt_t2i, test_dataset.img2txt, test_dataset.txt2img, args.filename)
    result_npr = itm_eval(score_npr_i2t, score_npr_t2i, test_dataset.img2txt, test_dataset.txt2img, args.filename)
    print("result",result)
    print("result_adv",result_adv)
    print("result_jpeg",result_jpeg)
    print("result_bit",result_bit)
    print("result_lt",result_lt)
    print("result_npr",result_npr)
    log_stats = {**{f'org ------ test_{k}': v for k, v in result.items()},
                 'eval type': args.adv, 'cls':args.cls, 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha}
    log_stats_adv = {**{f'adv ------ test_{k}': v for k, v in result_adv.items()},
                 'eval type': args.adv, 'cls':args.cls, 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha}
    with open(os.path.join(args.output_dir, "log_CLIP.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")
        f.write(json.dumps(log_stats_adv) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--filename', default='')
    
    parser.add_argument('--output_dir', default='output/retrieval')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--image_encoder', default='ViT-B/16')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=0, type=int,
                        help='0=clean, 1=adv text, 2=adv image, 3=adv sep, 4=adv co, 5=adv sl')
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--cls', action='store_true')

    args = parser.parse_args()

    # the output of CLIP is [CLS] embedding, so needn't to select at 0
    args.cls = False
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

