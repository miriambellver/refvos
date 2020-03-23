import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

import h5py

from refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 input_size,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 use_embeddings=True,
                 embedding_option=None,
                 eval_mode=False):

        self.max_seq_len = args.gt_maxseqlen
        self.classes = []
        self.input_size = input_size
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.data_root, args.refer_dataset, args.splitBy)
        self.mode = mode

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []

        self.eval_mode = eval_mode

        self.glove_table = {}

        with open(args.glove_dict, mode='r') as f:
            for line in f:
                words = line.split()

                in_w = len(words) - 300
                self.glove_table[' '.join(words[:in_w])] = [float(w) for w in words[-300:]]

        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            padding_glove = [0] * 300

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens

                # truncation of tokens
                input_ids = sentence_raw[:self.max_tokens]
                embs = []

                len_tokens = 0

                for w in input_ids:
                    if w in self.glove_table:
                        glove_emb = self.glove_table[w]
                        embs.append(glove_emb)
                        len_tokens += 1

                padded_w = self.max_tokens - len_tokens
                
                for p in range(padded_w):
                    embs.append(padding_glove)

                attention_mask[:len_tokens] = [1]*len_tokens


                sentences_for_ref.append(torch.tensor(embs).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))


            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

        self.use_embeddings = use_embeddings

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):

        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        this_sent_ids = ref[0]['sent_ids']

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:

            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]

                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))


            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)

        else:

            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask

