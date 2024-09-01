import re
from pathlib import Path

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from torch.utils.data import DataLoader, Dataset


class CustomImageDataset(Dataset):
     def __init__(self, image_paths, feature_extractor):
         self.image_paths = image_paths
         self.feature_extractor = feature_extractor
     
     def __len__(self):
         return len(self.image_paths)
     
     def __getitem__(self, idx):
         img_paths = [self.image_paths[idx]]
         new_imgs = []
         for img_or_path in img_paths:
            if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
                img = Image.open(img_or_path)
            elif isinstance(img_or_path, Image.Image):
                img = img_or_path
            else:
                raise ValueError(f'Invalid value of img_or_path: {img_or_path}')
            new_imgs.append(img)
         image_paths = new_imgs
         imgs = [img.convert('L').convert('RGB') for img in image_paths]
         image = self.feature_extractor(images=imgs, return_tensors="pt").pixel_values
         image = image.squeeze()
         return image


class MangaOcr:
    def __init__(self, pretrained_model_name_or_path='kha-white/manga-ocr-base', force_cpu=False, batch_size=16):
        logger.info(f'Loading OCR model from {pretrained_model_name_or_path}')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        logger.info(f'batch size: {batch_size}')
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
        self.batch_size = batch_size

        if not force_cpu and torch.cuda.is_available():
            logger.info('Using CUDA')
            self.model.cuda()
        else:
            logger.info('Using CPU')

        logger.info('OCR ready')

    def __call__(self, imgs_or_paths):
        if not isinstance(imgs_or_paths, list):
            imgs_or_paths = [imgs_or_paths]

        dataset = CustomImageDataset(imgs_or_paths, self.feature_extractor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
 

        images = []
        for img_or_path in imgs_or_paths:
            if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
                img = Image.open(img_or_path)
            elif isinstance(img_or_path, Image.Image):
                img = img_or_path
            else:
                raise ValueError(f'Invalid value of img_or_path: {img_or_path}')
            img = img.convert('L').convert('RGB')
            images.append(img)

        results = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch.to(self.model.device)
                # Generate captions
                outputs = self.model.generate(pixel_values=images, max_length=1000)

                # Decode the generated captions
                batch_captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(batch_captions)

        # results = []
        # for i in range(0, len(images), self.batch_size):
        #     batch = images[i:i + self.batch_size]
        #     pixel_values = self._preprocess(batch)
        #     generated_ids = self.model.generate(pixel_values.to(self.model.device), batch_size=self.batch_size)
        #     texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        #     texts = [post_process(text) for text in texts]
        #     results.extend(texts)

        return results

    def _preprocess(self, imgs):
        pixel_values = self.feature_extractor(imgs, return_tensors="pt").pixel_values
        return pixel_values


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
