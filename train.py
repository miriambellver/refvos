import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce #python 3
import operator

from pytorch_transformers import *

import torchvision

from coco_utils import get_coco
import transforms as T
import utils

import numpy as np

# to visualize the curves
from logger import Logger

import gc


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
                          input_size=(256, 448))

        num_classes = 2

    elif name == 'a2d':

        from data.a2d import A2DDataset

        ds = A2DDataset(args,
                        train= image_set == 'train',
                        db_root_dir= args.a2d_root_dir,
                        transform=transform,
                        inputRes=(args.size_a2d_x, args.size_a2d_y))

        num_classes = 2

    elif name == 'davis':

        from data.davis2017_4 import DAVIS17Offline

        ds = DAVIS17(train= image_set == 'train',
            db_root_dir='/gpfs/scratch/bsc31/bsc31429/dev/vog/datasets/DAVIS2017/',
            transform=transform,
            emb_type=emb_type)

        num_classes = 2


    return ds, num_classes


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr - args.lr_specific_decrease*epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# IoU calculation for proper validation
def IoU(pred, gt):

    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou


def get_transform(train, base_size=520, crop_size=480):

    min_size = int((0.8 if train else 1.0) * base_size)
    max_size = int((0.8 if train else 1.0) * base_size)

    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))

    if train:
        transforms.append(T.RandomCrop(crop_size))


    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)



def criterion(inputs, target, args):
    losses = {}
    for name, x in inputs.items():

        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, args, bert_model, device, num_classes, epoch, logger, baseline_model):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    val_loss = 0
    seg_loss = 0
    cos_loss = 0
    total_its = 0

    acc_ious = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):

            total_its += 1

            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), sentences.to(
                        device), attentions.to(device)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)


            if args.baseline_bilstm:

                num_tokens = torch.sum(attentions, dim=-1)

                unbinded_sequences = list(torch.unbind(sentences, dim=0))
                processed_seqs = [seq[:num_tokens[i], :] for i, seq in enumerate(unbinded_sequences)]

                packed_sentences = torch.nn.utils.rnn.pack_sequence(processed_seqs, enforce_sorted=False)

                hidden_states, cell_states = baseline_model[0](packed_sentences)
                hidden_states = torch.nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True,
                                                                       total_length=20)
                hidden_states = hidden_states[0]
                unbinded_hidden_states = list(torch.unbind(hidden_states, dim=0))

                processed_hidden_states = [seq[:num_tokens[i], :] for i, seq in enumerate(unbinded_hidden_states)]

                mean_hidden_states = [torch.mean(seq, dim=0).unsqueeze(0) for seq in processed_hidden_states]
                last_hidden_states = torch.cat(mean_hidden_states, dim=0)


                last_hidden_states = baseline_model[1](last_hidden_states)
                last_hidden_states = last_hidden_states.unsqueeze(1)

            else:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]


            embedding = last_hidden_states[:, 0, :]
            output, vis_emb, lan_emb = model(image, embedding.squeeze(1))

            iou = IoU(output['out'], target)
            acc_ious += iou

            loss = criterion(output, target, args)

            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

        val_loss = val_loss/total_its
        iou = acc_ious / total_its

        logger.scalar_summary('loss', val_loss, epoch)
        logger.scalar_summary('iou', iou, epoch)


    return confmat, iou


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args, print_freq, logger,
                    iterations, bert_model, baseline_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0
    train_emb_loss = 0
    train_seg_loss = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):

        total_its += 1

        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.to(device), target.to(device), sentences.to(device), attentions.to(device)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if args.baseline_bilstm:
            
            num_tokens = torch.sum(attentions, dim=-1)
            unbinded_sequences = list(torch.unbind(sentences, dim=0))
            processed_seqs = [seq[:num_tokens[i], :] for i, seq in enumerate(unbinded_sequences)]

            packed_sentences = torch.nn.utils.rnn.pack_sequence(processed_seqs, enforce_sorted=False)
            hidden_states, cell_states = baseline_model[0](packed_sentences)
            hidden_states = torch.nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True, total_length=20)

            hidden_states = hidden_states[0]

            unbinded_hidden_states = list(torch.unbind(hidden_states, dim=0))

            processed_hidden_states = [seq[:num_tokens[i], :] for i, seq in enumerate(unbinded_hidden_states)]

            mean_hidden_states = [torch.mean(seq, dim=0).unsqueeze(0) for seq in processed_hidden_states]
            last_hidden_states = torch.cat(mean_hidden_states, dim=0)


            last_hidden_states = baseline_model[1](last_hidden_states)
            last_hidden_states = last_hidden_states.unsqueeze(1)

        else:

            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
             
        embedding = last_hidden_states[:, 0, :]
        output, vis_emb, lan_emb = model(image, embedding.squeeze(1))

        loss = criterion(output, target, args)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.linear_lr:
            adjust_learning_rate(optimizer, epoch, args)
        else:
            lr_scheduler.step()

        train_loss += loss.item()
        iterations += 1

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, embedding, output, vis_emb, lan_emb, last_hidden_states, data

        gc.collect()
        torch.cuda.empty_cache()

    train_loss = train_loss/total_its

    logger.scalar_summary('loss', train_loss, epoch)
    logger.scalar_summary('lr', optimizer.param_groups[0]["lr"], epoch)


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.dataset, "train",
        get_transform(train=True, base_size=args.base_size, crop_size=args.crop_size), args=args)
    
    dataset_test, _ = get_dataset(args.dataset, "val",
        get_transform(train=False, base_size=args.base_size, crop_size=args.crop_size), args=args)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn_emb_berts, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn_emb_berts)

    model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,
                                                                 aux_loss=args.aux_loss,
                                                                 pretrained=args.pretrained,
                                                                 embedding_model=args.embedding_model,
                                                                 aspp_option=args.aspp_option,
                                                                 args=args)

    model_class = BertModel
    bert_model = model_class.from_pretrained(args.ck_bert)

    if args.baseline_bilstm:

        bilstm = torch.nn.LSTM(input_size=300, hidden_size=1000, num_layers=1, bidirectional=True, batch_first=True)
        fc_layer = torch.nn.Linear(2000, 768)
        bilstm = bilstm.cuda()
        fc_layer = fc_layer.cuda()

    if args.pretrained_refcoco:

        checkpoint = torch.load(args.ck_pretrained_coco)
        model.load_state_dict(checkpoint['model'])
        bert_model.load_state_dict(checkpoint['bert_model'])

    elif args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        if args.baseline_bilstm:
            bilstm.load_state_dict(checkpoint['bilstm'])
            fc_layer.load_state_dict(checkpoint['fc_layer'])

    model_without_ddp = model
    bert_model_without_ddp = bert_model

    if args.test_only:
        confmat = evaluate(model, data_loader_test, args, bert_model, epoch=0, device=device, num_classes=num_classes, baseline_model=[bilstm, fc_layer])
        print(confmat)
        return


    if args.baseline_bilstm:

        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},

            {"params": [p for p in bilstm.parameters() if p.requires_grad]},
            {"params": [p for p in fc_layer.parameters() if p.requires_grad]}
        ]

    else:

        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat, [[p for p in bert_model_without_ddp.encoder.layer[i].parameters() if p.requires_grad] for i in range(10)])},
            {"params": [p for p in bert_model_without_ddp.pooler.parameters() if p.requires_grad]}
        ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.fixed_lr:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: args.lr_specific)
    elif args.linear_lr:
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    model_dir = os.path.join('./models/', args.model_id)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, 'train'))
        os.makedirs(os.path.join(model_dir, 'val'))

    logger_train = Logger(os.path.join(model_dir, 'train'))
    logger_val = Logger(os.path.join(model_dir, 'val'))

    start_time = time.time()

    iterations = 0
    t_iou = 0

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

        if not args.fixed_lr:
            if not args.linear_lr:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.baseline_bilstm:
        baseline_model = [bilstm, fc_layer]
    else:
        baseline_model = None

    for epoch in range(args.epochs):
        
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args, args.print_freq,
                        logger_train, iterations, bert_model, baseline_model=baseline_model)

        confmat, iou = evaluate(model, data_loader_test, args, bert_model, epoch=epoch, device=device,
                                num_classes=num_classes, logger=logger_val, baseline_model=baseline_model)

        print(confmat)


        # only save if checkpoint improves
        if t_iou < iou:
            print('Better epoch!_{}\n'.format(epoch))

            if args.baseline_bilstm:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'bilstm': bilstm.state_dict(),
                        'fc_layer': fc_layer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'lr_scheduler': lr_scheduler.state_dict()
                    },
                    os.path.join(args.output_dir, 'model_best_{}.pth'.format(args.model_id)))

            else:
                dict_to_save = {'model': model_without_ddp.state_dict(),
                'bert_model': bert_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args}

                if not args.linear_lr:
                    dict_to_save['lr_scheduler'] = lr_scheduler.state_dict()
                
                utils.save_on_master(dict_to_save, os.path.join(args.output_dir, 'model_best_{}.pth'.format(args.model_id)))

            t_iou = iou

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()

    main(args)

