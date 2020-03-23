import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--model_id', default='my_model', help='name to identify model')

    parser.add_argument('--dataset', default='refcoco', help='the different options are refcoco, davis or a2d')
    parser.add_argument('--model', default='deeplabv3_resnet101', help='model')

    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')

    parser.add_argument('-b', '--batch-size', default=8, type=int)

    parser.add_argument('--base_size', default=520, type=int, help='base_size')
    parser.add_argument('--crop_size', default=480, type=int, help='crop_size')

    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true",)

    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/gpfs/scratch/bsc31/bsc31429/dev/vog/baseline_refcoco/checkpoints/', help='path where to save checkpoints')

    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true",)

    # Fusion language + visual
    parser.add_argument('--multiply_feats', action='store_true', help='multiplication of features instead of concatanation')
    parser.add_argument('--addition',  action='store_true', help='residual connection when features fusion')

    # Learning rate strategies
    parser.add_argument('--fixed_lr', action='store_true', help='whether to fix the learning rate')
    parser.add_argument('--linear_lr', action='store_true', help='lr scheduler 5')
    parser.add_argument('--lr_specific', default=0.00003, type=float, help='specific lr for fixed configuration')
    parser.add_argument('--lr_specific_decrease', default=0.001, type=float, help='specific lr for fixed configuration')
    
    
    #### Baseline 
    parser.add_argument('--baseline_bilstm', action='store_true', help='baseline bidirectional LSTM')


    #### Training configurations
    parser.add_argument('--load_optimizer', action='store_true', help='load optimizer')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--bert_tokenizer',  default='/gpfs/scratch/bsc31/bsc31429/dev/vog/pytorch-transformers/bert-base-uncased-vocab.txt', help='tokenizer BERT')
    parser.add_argument('--glove_dict',  default='/gpfs/scratch/bsc31/bsc31429/dev/vog/glove.840B.300d.txt', help='glove dict')

    #### Testing parameters
    parser.add_argument('--results_folder',  default='/gpfs/scratch/bsc31/bsc31429/dev/refvos/results/', help='glove dict')
    parser.add_argument('--submission_path',  default='/gpfs/scratch/bsc31/bsc31429/dev/refvos/results_davis_full_video_ann1_submission/', help='glove dict')

    #### Dataset specifics

    # pretraining
    parser.add_argument("--pretrained_refcoco", dest="pretrained_refcoco", help="Use pre-trained models from refcoco", action="store_true",)
    parser.add_argument('--ck_pretrained_coco',  default='/gpfs/scratch/bsc31/bsc31429/dev/vog/baseline_refcoco/checkpoints/model_refcoco_v4_bert_all_encoder_14.pth', help='which CK pretrained on refcoco to consider')

    # REFER
    parser.add_argument('--data_root', default='/gpfs/scratch/bsc31/bsc31429/dev/vog/datasets/refer/data/', help='dataset root directory')
    parser.add_argument('--refer_dataset', default='refcoco', help='dataset name')
    parser.add_argument('--splitBy', default='unc', help='split By')

    # DAVIS
    parser.add_argument('--emb_type', default='first_mask', help='first_mask or full_video for davis')
    parser.add_argument('--davis_data_root', default='/gpfs/scratch/bsc31/bsc31429/databases/davis2017/DAVIS', help='davis dataset root directory')
    parser.add_argument('--davis_annotations_file', default='/gpfs/scratch/bsc31/bsc31429/databases/davis2017/davis_text_annotations/Davis17_annot1_full_video.txt', help='path of annotations file')

    # A2D
    parser.add_argument('--a2d_root_dir', default='/gpfs/scratch/bsc31/bsc31429/databases/Release/')
    parser.add_argument('--size_a2d_x', default=240, type=int, help='x size for a2d images')
    parser.add_argument('--size_a2d_y', default=427, type=int, help='y size for a2d images')
    parser.add_argument('--a2d_annotations_file', default='/gpfs/scratch/bsc31/bsc31429/databases/Release/a2d_annotation.txt',  help='path of annotations file')


    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()