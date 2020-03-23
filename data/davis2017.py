"""
-Author: Omkar Damle
-Date: April 2018

Modified Version:
-Author: Miriam Bellver
-Date: April 2020
"""
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image
import cv2

import torch
import h5py

import transformers


class DAVIS17(Dataset):
    def __init__(self, args,
                train=True,
                db_root_dir='DAVIS17',
                transform=None,
                emb_type='first_mask',
                annotator=1):

        self.train = train
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.annotator = annotator

        if self.train:
            fname = 'train'
        else:
            fname = 'val'
      
        img_list = []
        labels = []
        objs = []
        ids = []

        max_tokens = 10

        self.input_ids = []
        self.attention_masks = []

        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_tokenizer)

        data_root = os.path.join(db_root_dir, 'davis_text_annotations/')

        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[PAD]'))


        if emb_type == 'first_mask':
            files = ['Davis17_annot1.txt', 'Davis17_annot2.txt']
        else:
            files = ['Davis17_annot1_full_video.txt', 'Davis17_annot2_full_video.txt']

        # read annotations from annotator 1
        with open(os.path.join(data_root, files[0]), mode='r') as a:
            lines1 = a.readlines()

        # read annotations from annotator 2
        with open(os.path.join(data_root, files[1]), mode='r', encoding="latin-1") as a:
            lines2 = a.readlines()

        annotations = {}

        for l1, l2 in zip(lines1, lines2):

            words1 = l1.split()
            words2 = l2.split()

            sentences = [words1, words2]

            for i, s in enumerate(sentences):
                raw_s = ' '.join(s[2:])[1:-1]
                annotations[s[0] + '_' + str(int(s[1])-1) + '_' + str(i)] = raw_s

        # Initialize the original DAVIS splits for training the parent network
        with open(os.path.join(db_root_dir, 'DAVIS/ImageSets/2017/' + fname + '.txt')) as f:
            seqs = f.readlines()

            for seq in seqs:
                images = np.sort(os.listdir(os.path.join(db_root_dir, 'DAVIS/JPEGImages/480p/', seq.strip())))

                image_id_first_frame = images[0].split('.')[0]

                # check number of objects
                annot_path = os.path.join('DAVIS/Annotations/480p', seq.strip(), image_id_first_frame + '.png')
                annot = np.asarray(Image.open(os.path.join(self.db_root_dir, annot_path)))
                # determine number of objects by first frame TODO: For full video annotations this is not necessary
                num_objs = len(np.unique(annot)) - 1

                for j, image in enumerate(images):

                    image_id = image.split('.')[0]

                    for i in range(num_objs):

                        img_list.append(os.path.join('DAVIS/JPEGImages/480p', seq.strip(), image))
                        labels.append(os.path.join('DAVIS/Annotations/480p', seq.strip(), image_id + '.png'))

                        objs.append(i+1)
                        ids.append(seq.split('\n')[0] + '_' + str(i+1))

                        sentences_for_ref = []
                        attentions_for_ref = []

                        # consider which annotator
                        for l in range(2):

                            if seq.strip() + '_' + str(i) + '_' + str(l) in annotations:
                                sentence_raw = annotations[seq.strip() + '_' + str(i) + '_' + str(l)]

                                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                                input_ids = input_ids[:max_tokens]
                                attention_mask = [0] * max_tokens
                                padded_input_ids = [0] * max_tokens

                                padded_input_ids[:len(input_ids)] = input_ids
                                attention_mask[:len(input_ids)] = [1] * len(input_ids)

                                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

                        self.input_ids.append(sentences_for_ref)
                        self.attention_masks.append(attentions_for_ref)

        self.img_list = img_list
        self.labels = labels
        self.objs = objs
        self.ids = ids

        print('Done initializing ' + fname + ' Dataset')

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.db_root_dir, self.img_list[idx]))
        label = np.array(Image.open(os.path.join(self.db_root_dir, self.labels[idx])))

        obj_id = self.objs[idx]

        mask = (label == obj_id).astype(np.float32)
        label = Image.fromarray(mask).convert('L')

        img, label = self.transform(img, label)

        # in case it is test time, we choose which annotator to consider
        if self.train == False:
            choice_sent = self.annotator - 1

        else:
            choice_sent = np.random.choice(len(self.input_ids[idx]))

        emb = self.input_ids[idx][choice_sent]
        attention_mask = self.attention_masks[idx][choice_sent]

        return img, label, emb, attention_mask

    def __len__(self):
        return len(self.img_list)
 
