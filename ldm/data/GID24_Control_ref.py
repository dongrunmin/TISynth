import os
from re import L
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json

import random

# Some words may differ from the class names defined in GID24 to minimize ambiguity
GID24_dict = {
'0':'industrial land',
'1':'paddy field',
'2':'irrigated field',
'3':'dry cropland',
'4':'garden land',
'5':'arbor forest',
'6':'shrub forest',
'7':'park land',
'8':'natural meadow',
'9':'artificial meadow',
'10':'river',
'11':'urban residential',
'12':'lake',
'13':'pond',
'14':'fish pond',
'15':'snow',
'16':'bareland',
'17':'rural residential',
'18':'stadium',
'19':'square',
'20':'road',
'21':'overpass',
'22':'railway station',
'23':'airport'
}


class GID24Base(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        self.data = []
        #self.data_paths=txt_file=prompt.json
        with open(self.data_paths, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))
        self._length = len(self.data)
        self.size = size
        self.interpolation = {"nearest": Image.NEAREST,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        path = os.path.join(self.data_root, target_filename)
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        path_ = path[:-4]
        path2 = os.path.join(self.data_root, source_filename)
        pil_image2 = Image.open(path2)

        flip = random.random() < self.flip_p

        ssl_image = pil_image.copy()

        if self.size is not None:
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            pil_image2 = pil_image2.resize((self.size, self.size), resample=Image.NEAREST)
        if flip:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            pil_image2 = pil_image2.transpose(Image.FLIP_LEFT_RIGHT)
            ssl_image = ssl_image.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.array(pil_image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        ssl_image = np.array(ssl_image).astype(np.uint8)
        ssl_image = (ssl_image / 127.5 - 1.0).astype(np.float32)

        label = np.array(pil_image2).astype(np.float32)
        if label.ndim == 2:
            label = np.expand_dims(label, 2).repeat(3, 2)
        # Normalize source images to [0, 1]
        label = label / 255.0

        return dict(jpg=image, txt=prompt, hint=label, ssl=ssl_image)



class GID24Train(GID24Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GID24Validation(GID24Base):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
