
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from PIL import Image
import cv2

import torch
import h5py

from pytorch_transformers import *

import csv


class A2DDataset(Dataset):
    def __init__(self, 
        args, train=True, 
        inputRes=None,
        transform=None):

        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.max_tokens = 10
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[PAD]'))
        annotations_file = args.a2d_annotations_file

        # read annotations from annotator
        with open(annotations_file, mode='r') as a:
            lines = a.readlines()

        annotations_ref = {}
        for l in lines:

            words = l.split(',')

            if len(words) > 1:

                if not words[0] in annotations_ref and words[0]:
                    annotations_ref[words[0]] = {}

                annotations_ref[words[0]][words[1]] = {}
                annotations_ref[words[0]][words[1]] = words[2].split('\n')[0]


        if train:
            id_split = 0
            fname = 'train'
        else:
            id_split = 1
            fname = 'val'

        ids_list = []
        img_list = []
        labels = []
        annots_info_list = []
        objs = []
        num_objs_list = []
        ids = []
        sentences = []
        attentions = []

        raw_sentences = []

        with open(os.path.join(db_root_dir, 'videoset.csv')) as f:

            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:

                seq = row[0]

                if int(row[-1]) == id_split and seq in annotations_ref:

                    annotations = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/col', seq)))
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'pngs320H/', seq.strip())))

                    image_id_first_frame = annotations[0].split('.')[0]
                    # check number of objects
                    annot_info_path = os.path.join('a2d_annotation_with_instances', seq.strip(), image_id_first_frame + '.h5')
                    annot_info = h5py.File(os.path.join(self.db_root_dir, annot_info_path), 'r')
                    num_objs = len(list(annot_info['instance']))

                    max_num_objs = 0

                    for p in range(len(annotations)):

                        annot_info_path = os.path.join('a2d_annotation_with_instances', seq.strip(),
                                                       annotations[p].split('.')[0] + '.h5')
                        annot_info = h5py.File(os.path.join(self.db_root_dir, annot_info_path), 'r')

                        num_objs_ind = max(list(annot_info['instance'])) + 1
                       
                        if max_num_objs < num_objs_ind:
                            max_num_objs = num_objs_ind

                    num_objs = max_num_objs

                    for i in range(num_objs):

                        sentences_for_ref = []
                        attentions_for_ref = []

                        # is this a mistake of the dataset???
                        if seq == 'LmMXXT4DfWk' and i == 4:
                            sentence_raw = annotations_ref[seq]['5']

                        else:
                            sentence_raw = annotations_ref[seq][str(i)]

                        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                        input_ids = input_ids[:self.max_tokens]

                        attention_mask = [0] * self.max_tokens
                        padded_input_ids = [0] * self.max_tokens

                        padded_input_ids[:len(input_ids)] = input_ids
                        attention_mask[:len(input_ids)] = [1] * len(input_ids)

                        sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                        attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

                        for j in range(len(annotations)):

                            image_id = annotations[j].split('.')[0]

                            raw_sentences.append(sentence_raw)

                            img_list.append(os.path.join('pngs320H', seq.strip(), image_id + '.png'))

                            annots_info_list.append(os.path.join('a2d_annotation_with_instances', seq.strip(), image_id + '.h5'))
                            labels.append(os.path.join('Annotations/col', seq.strip(), image_id + '.png'))

                            objs.append(i)
                            ids.append(seq.split('\n')[0] + '_' + str(i))

                            sentences.append(sentences_for_ref)
                            attentions.append(attentions_for_ref)

                            pseudo_annots.append(0)

        self.img_list = img_list
        self.objs = objs
        self.num_objs_list = num_objs_list
        self.labels = labels
        self.sentences = sentences
        self.attentions = attentions
        self.annots_info_list = annots_info_list
        self.raw_sentences = raw_sentences

        print('Done initializing ' + fname + ' Dataset')

    def __getitem__(self, idx):

        this_img = self.img_list[idx]

        img = Image.open(os.path.join(self.db_root_dir, this_img))
        obj_id = self.objs[idx]
       
        a_info = h5py.File(os.path.join(self.db_root_dir, self.annots_info_list[idx]), 'r')
        pos = np.where(np.array(list(a_info['instance'])) == obj_id)

        if len(pos[0]) == 0:
            mask = np.zeros([np.asarray(img).shape[1], np.asarray(img).shape[0]])
        else:
            pos = int(pos[0][0])

            if len(np.array(a_info['reMask']).shape) == 3:
                mask = a_info['reMask'][pos, :, :]
            else:
                mask = np.array(a_info['reMask'])

        mask = np.transpose(mask, (1, 0))

        label = Image.fromarray(mask).convert('L')
        img, label = self.transform(img, label)

        choice_id = np.random.choice(len(self.sentences[idx]))

        sentences = self.sentences[idx][choice_id]
        attentions = self.attentions[idx][choice_id]

        return img, label, sentences, attentions

    def __len__(self):
        return len(self.img_list)





