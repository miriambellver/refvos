import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from transformers import *
import torchvision

from lib import segmentation

import transforms as T
import utils

import numpy as np

from pycocotools import mask
from scipy.misc import imread


def get_dataset(name, image_set, transform, args):

    if name == 'refcoco' or name == 'refcoco+':

        if args.baseline_bilstm:
            from data.dataset_refer_glove import ReferDataset
        else:
            from data.dataset_refer_bert import ReferDataset

        ds = ReferDataset(args,
                          split=image_set,
                          image_transforms=transform,
                          target_transforms=None,
                          input_size=(256, 448),
                          eval_mode=True)

        num_classes = 2

    elif name == 'a2d':

        from data.a2d import A2DDataset

        ds = A2DDataset(args,
                        train= image_set == 'train',
                        db_root_dir= args.a2d_data_root,
                        transform=transform,
                        inputRes=(args.size_a2d_x, args.size_a2d_y))

        num_classes = 2

    elif name == 'davis':

        from data.davis2017 import DAVIS17

        ds = DAVIS17(args,
                    train= image_set == 'train',
                    db_root_dir=args.davis_data_root,
                    transform=transform,
                    emb_type=args.emb_type)

        num_classes = 2

    return ds, num_classes


def evaluate(args, model, data_loader, ref_ids, refer, bert_model, device, num_classes, display=False, baseline_model=None, 
    objs_ids=None, num_objs_list=None):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    refs_ids_list = []
    outputs = []
    # dict to save results for DAVIS
    total_outputs = {}

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []


    header = 'Test:'
    with torch.no_grad():
        k = 0
        l = 0
        for image, target, sentences, attentions in metric_logger.log_every(data_loader, 100, header):

            image, target, sentences, attentions = image.to(device), target.to(device), sentences.to(device), attentions.to(device)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if args.dataset == 'davis' or args.dataset == 'a2d':
                sentences = sentences.unsqueeze(-1)
                attentions = attentions.unsqueeze(-1)

            target = target.cpu().data.numpy()

                
            for j in range(sentences.size(-1)):

                refs_ids_list.append(k)

                if args.baseline_bilstm:

                    sent = sentences[:, :, :, j]
                    att = attentions[:, :, j]

                    num_tokens = torch.sum(att, dim=-1)
                    processed_seqs = sent[:num_tokens, :]

                    hidden_states, cell_states = baseline_model[0](processed_seqs)
                    hidden_states = hidden_states[0]

                    processed_hidden_states = hidden_states[:num_tokens, :]

                    last_hidden_states = torch.mean(processed_hidden_states, dim=0)

                    last_hidden_states = baseline_model[1](last_hidden_states)
                    embedding = last_hidden_states.unsqueeze(1)

                else:

                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                
                embedding = last_hidden_states[:, 0, :]

                output, _, _ = model(image, embedding.squeeze(1))

                output = output['out'].cpu()
                output_mask = output.argmax(1).data.numpy()
                outputs.append(output_mask)

                I, U = computeIoU(output_mask, target)

                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U

                mean_IoU.append(this_iou)

                cum_I += I
                cum_U += U

                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1


            del image, target, attentions

            if display:

                plt.figure()
                plt.axis('off')

                if args.dataset == 'refcoco' or args.dataset == 'refcoco+':
                    ref = refer.loadRefs(ref_ids[k])
                    image_info = refer.Imgs[ref[0]['image_id']]

                for p in range(len(ref[0]['sentences'])):

                    l += 1

                    if  args.dataset == 'refcoco' or args.dataset == 'refcoco+':
                        sentence = ref[0]['sentences'][p]['raw']
                        im_path = os.path.join(refer.IMAGE_DIR, image_info['file_name'])
                    elif args.dataset == 'davis':
                        idx = ref_ids[k]
                        sentence = refer[idx]
                        im_path = os.path.join(args.davis_data_root, img_list[k])
                    elif args.dataset == 'a2d':
                        sentence = refer[k]
                        image_name = ref_ids[k]
                        im_path = os.path.join(args.a2d_root_dir, image_name)


                    im = imread(im_path)
                    plt.imshow(im)

                    if args.dataset == 'davis':
                        if img_list[k] not in total_outputs:
                            total_outputs[img_list[k]] = {}
                            o_mask = output_mask.copy()
                            o_mask = o_mask.astype(int)*int(idx.split('_')[-1])
                            total_outputs[img_list[k]] = o_mask.squeeze(0)
                        else:
                            total_outputs[img_list[k]][output_mask.squeeze(0) == True] = int(idx.split('_')[-1])

                    plt.text(0, 0, sentence, fontsize=12)

                    ax = plt.gca()
                    ax.set_autoscale_on(False)

                    # mask definition
                    img = np.ones((im.shape[0], im.shape[1], 3))
                    color_mask = np.array([0, 255, 0]) / 255.0
                    for i in range(3):
                        img[:, :, i] = color_mask[i]

                    if  args.dataset == 'refcoco' or args.dataset == 'refcoco+':
                        output_mask = outputs[-len(ref[0]['sentences'])+p].transpose(1, 2, 0)

                    ax.imshow(np.dstack((img, output_mask * 0.5)))

                    if not os.path.isdir(results_folder):
                        os.makedirs(results_folder)

                    figname = os.path.join(args.results_folder, str(l) + '.png')
                    plt.close()

            k += 1


        if args.dataset == 'davis':

            for r in total_outputs.keys():

                new_im = Image.fromarray(total_outputs[r].astype(np.uint8))
                file_name = r.split('/')[-1].split('.')[0]
                folder_name = r.split('/')[-2]

                if not os.path.isdir(os.path.join(submission_path, folder_name)):
                    os.makedirs(os.path.join(submission_path, folder_name))

                new_im.save(os.path.join(args.submission_path, folder_name, file_name + '.png'))

  
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)

    print(results_str)

    return refs_ids_list, outputs


def get_transform():
    
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


# compute IoU
def computeIoU(pred_seg, gd_seg):

    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):

    device = torch.device(args.device)

    dataset_test, _ = get_dataset(args.dataset, args.split, get_transform(), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler,
                                                   num_workers=args.workers, collate_fn=utils.collate_fn_emb_berts)

    model = segmentation.__dict__[args.model](num_classes=2,
        aux_loss=False,
        pretrained=False,
        args=args)

    model.to(device)
    model_class = BertModel

    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.to(device)


    if args.baseline_bilstm:
        bilstm = torch.nn.LSTM(input_size=300, hidden_size=1000, num_layers=1, bidirectional=True, batch_first=True)
        fc_layer = torch.nn.Linear(2000, 768)
        bilstm = bilstm.to(device)
        fc_layer = fc_layer.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')

    bert_model.load_state_dict(checkpoint['bert_model'])
    model.load_state_dict(checkpoint['model'])

    if args.baseline_bilstm:
        bilstm.load_state_dict(checkpoint['bilstm'])
        fc_layer.load_state_dict(checkpoint['fc_layer'])

    if args.dataset == 'refcoco' or args.dataset == 'refcoco+':
        ref_ids = dataset_test.ref_ids
        refer = dataset_test.refer
        ids = ref_ids
        objs_ids = None
        num_objs_list = None
    elif args.dataset == 'davis':
        ids = dataset_test.ids
        objs_ids = None
        num_objs_list = None

        with open(args.davis_annotations_file) as f:
            lines = f.readlines()

    elif args.dataset == 'a2d':
        ids = dataset_test.img_list
        objs_ids = dataset_test.objs
        num_objs_list = dataset_test.num_objs_list

        with open(args.davis_annotations_file) as f:
            lines = f.readlines()

    if args.dataset == 'davis' or args.dataset == 'a2d':

        refer = {}

        for l in lines:
            words = l.split()

            refer[words[0] + '_' + words[1]] = {}
            refer[words[0] + '_' + words[1]] = ' '.join(words[2:])[1:-1]

    if args.baseline_bilstm:
        baseline_model = [bilstm, fc_layer]
    else:
        baseline_model = None

    refs_ids_list, outputs = evaluate(args, model, data_loader_test, ids, refer, bert_model, device=device, 
        num_classes=2, baseline_model=baseline_model,  objs_ids=objs_ids, num_objs_list=num_objs_list)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
