import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='RefVOS Training')

    parser.add_argument('--model_id', default='my_model', help='name to identify model')

    parser.add_argument('--dataset', default='refcoco', help='choose one of the following datasets: refcoco, refcoco+, davis or a2d')
    parser.add_argument('--model', default='deeplabv3_resnet101', help='model')

    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')

    parser.add_argument('-b', '--batch-size', default=6, type=int)

    parser.add_argument('--base_size', default=520, type=int, help='base_size')
    parser.add_argument('--crop_size', default=480, type=int, help='crop_size')

    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true",)

    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoints')

    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true",)

    # Fusion language + visual
    parser.add_argument('--multiply_feats', action='store_true', default=True, help='multiplication of visual and language features')
    parser.add_argument('--addition',  action='store_true', help='addition of visual and language features')

    # Learning rate strategies
    parser.add_argument('--fixed_lr', action='store_true', help='use fixed learning rate')
    parser.add_argument('--linear_lr', action='store_true', help='use linear learning rate schedule')
    parser.add_argument('--lr_specific', default=0.00003, type=float, help='specific lr for fixed lr configuration')
    parser.add_argument('--lr_specific_decrease', default=0.001, type=float, help='specific lr decrease for linear lr configuration')
    
    
    #### Baseline 
    parser.add_argument('--baseline_bilstm', action='store_true', help='baseline bidirectional LSTM')


    #### Training configurations
    parser.add_argument('--load_optimizer', action='store_true', help='load optimizer')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--bert_tokenizer',  default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--glove_dict',  default='./glove.840B.300d.txt', help='glove dict that you need to download and save')
    parser.add_argument('--ck_bert',  default='bert-base-uncased', help='BERT pre-trained weights')


    #### Testing parameters
    parser.add_argument('--results_folder',  default='./results/', help='results folder')
    parser.add_argument('--submission_path',  default='./results_submission/', help='submission results folder for DAVIS')
    parser.add_argument('--split',  default='test', help='split to run test')
    parser.add_argument('--display',  action='store_true', help='save output predictions')

    
    #### Dataset specifics

    # pretraining
    parser.add_argument("--pretrained_refvos", dest="pretrained_refvos", help="Use pre-trained models for RefVOS", action="store_true",)
    parser.add_argument('--ck_pretrained_refvos',  default='./checkpoints/model_refcoco.pth', help='Pre-trained weights for RefVOS')

    # REFER
    parser.add_argument('--refer_data_root', default='./datasets/refer/data/', help='REFER dataset root directory')
    parser.add_argument('--refer_dataset', default='refcoco', help='dataset name')
    parser.add_argument('--splitBy', default='unc', help='split By')

    # DAVIS
    parser.add_argument('--davis_data_root', default='./datasets/davis2017', help='DAVIS dataset root directory')
    parser.add_argument('--davis_annotations_file', default='./datasets/davis2017/davis_text_annotations/Davis17_annot1_full_video.txt', help='path of DAVIS annotations file')
    parser.add_argument('--emb_type', default='first_mask', help='first_mask or full_video for DAVIS')

    # A2D
    parser.add_argument('--a2d_data_root', default='./datasets/Release/', help='A2D dataset root directory')
    parser.add_argument('--size_a2d_x', default=240, type=int, help='x size for A2D images')
    parser.add_argument('--size_a2d_y', default=427, type=int, help='y size for A2D images')
    parser.add_argument('--a2d_annotations_file', default='./datasets/Release/a2d_annotation.txt',  help='path of A2D annotations file')

    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
