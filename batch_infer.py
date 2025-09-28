import argparse, os
import torch
import einops
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from cldm.model import create_model, load_state_dict
from ldm.data.GID24_Control import GID24_dict
from cldm.ddim_hacked_ssl import DDIMSampler

from torch.utils.data import DataLoader, Dataset
import json
import time

class COCOVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'stuffthingmaps_trainval2017/val2017', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(182)
        text = ''
        for i in range(len(class_ids)):
            text += COCO_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

class ADE20KVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'annotations/validation', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(151)
        text = ''
        for i in range(len(class_ids)):
            text += ADE20K_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

class GID26KVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        self.data = []
        with open(self.data_paths, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))
        self._length = len(self.data)
        self.size = size
        self.interpolation = {"nearest": PIL.Image.NEAREST,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i]

        source_filename = item['source']
        img_name = source_filename.split('/')[-1][:-4]
        ref_filename = item['ref_image']
        prompt = item['prompt']

        ref_path = os.path.join(self.data_root, ref_filename)
        pil_ref_image = Image.open(ref_path)
        if not pil_ref_image.mode == "RGB":
            pil_ref_image = pil_ref_image.convert("RGB")
        ref_image = np.array(pil_ref_image).astype(np.uint8)
        ref_image = (ref_image / 127.5 - 1.0).astype(np.float32)
        #ref_image = ref_image / 255


        path2 = os.path.join(self.data_root, source_filename)
        pil_image2 = Image.open(path2)

        #image = np.array(pil_image).astype(np.uint8)
        #image = (image / 127.5 - 1.0).astype(np.float32)
        label = np.array(pil_image2).astype(np.float32)
        if label.ndim == 2:
            label = np.expand_dims(label, 2).repeat(3, 2)
        # Normalize source images to [0, 1]
        label = label / 255.0

        return dict(img_name=img_name, txt=prompt, hint=label, ref_image=ref_image)



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/layout2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-finetune_COCO.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True, 
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="path to txt file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="which dataset to evaluate",
        choices=["COCO", "ADE20K", "GID26K"],
        default="GID26K"
    )
    parser.add_argument(
        "--err_log",
        type=str,
        help="the path of error information",
        default="err_log"
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)
    model = create_model(opt.config).cpu()

    ##debug
    #for name in model.state_dict().keys():
    #    print(name)
    #print('_________________')
    #checkpoint = torch.load(opt.ckpt, map_location='cuda')
    #state_dict = checkpoint['state_dict']
    #for key in state_dict.keys():
    #    print(key)

    missing, _ = model.load_state_dict(load_state_dict(opt.ckpt, location='cuda'), strict=False)
    if missing:
        print(f"[WARN] Missing keys in state_dict: {missing}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size
    if opt.dataset == "GID26K":
        val_dataset = GID26KVal(data_root=opt.data_root, txt_file=opt.txt_file)

    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=batch_size, num_workers=2, shuffle=False)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    err_list  = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for idx, data in enumerate(val_dataloader):
                    st_time = time.perf_counter()
                    seed = str(opt.seed)
                    tmp_name = data["img_name"][-1]
                    tmp_path = os.path.join(outpath, f"{tmp_name}_seed_{seed}.jpg")
                    if os.path.exists(tmp_path):
                        continue
                        
                    BS = len(data['img_name'])
                    ref_latent = data["ref_image"].to(device)
                    ref_latent = einops.rearrange(ref_latent, 'b h w c -> b c h w').clone()
                    ref_latent = ref_latent.to(memory_format=torch.contiguous_format).float()
                    c_ssl = model.get_learned_ssl_conditioning(ref_latent).detach()

                    control = data["hint"].to(device)
                    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                    text = data["txt"]
                    c = model.get_learned_conditioning(text)
                    cond = {"c_concat": [control], "c_crossattn": [c], "c_ssl": [c_ssl]}
                    un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""] * BS)], "c_ssl": [c_ssl]}
                    shape = (opt.C, opt.H // opt.f, opt.W // opt.f)


                    strength = 1
                    guess_mode = False
                    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

                    scale = 9
                    samples, intermediates = sampler.sample(opt.ddim_steps, BS, 
                            shape, cond, verbose=False, eta=opt.ddim_eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)


                    x_samples_ddim = model.decode_first_stage(samples)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for i in range(len(x_samples_ddim)):
                        x_sample = x_samples_ddim[i]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img_name = data["img_name"][i]
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, f"{img_name}_seed_{seed}.jpg"))
                        
                    elapsed_hour = (time.perf_counter() - st_time) * (len(val_dataloader.dataset) / opt.batch_size - idx + 1) / 3600.0
                    print(f'\nRemain time: {elapsed_hour:.2f}h\n')
                    

if __name__ == "__main__":
    main()