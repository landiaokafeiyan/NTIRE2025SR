#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import argparse
import yaml
import logging
import torch
_logger = logging.getLogger('train')

def _parse_args():
    """ Parse command-line arguments and load config file if provided. """
    # Parse config file argument first
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                               help='YAML config file specifying default arguments')
    
    args_config, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # # Dataset parameters
    # group = parser.add_argument_group('Dataset parameters')
    # group.add_argument('--WEIGHT_L1', default=0.7, type=float, metavar='N', help='')
    # group.add_argument('--WEIGHT_PEC', default=0.3, type=float, metavar='N', help='')
    # # Keep this argument outside of the dataset group because it is positional.
    # parser.add_argument('--data_dir', metavar='DIR', default='dataset/train', 
    #                     help='path to train dataset')
    # # parser.add_argument('--data_dir', metavar='DIR', default='/scratch/ll5484/ntire25/dataset/LSDIR', 
    # #                     help='path to train dataset')
    # parser.add_argument('--dataset_name', metavar='DIR', default='DIV2K', help='path to train dataset')
    # parser.add_argument('--test_data_dir', metavar='DIR', default='dataset/val', 
    #                     help='path to dataset')
    # parser.add_argument('--json_path', metavar='DIR', default='utils/train_X4.json', 
    #                     help='path to dataset')
    # # parser.add_argument('--data_dir', metavar='DIR', default='images', 
    # #                     help='path to dataset')
    # group.add_argument('--dataset', '-d', metavar='NAME', default='ImageFolder/ImageTar',
    #                     help='dataset type (default: ImageFolder/ImageTar if empty)')
    # group.add_argument('--Notes', default='Test', type=str)
    # group.add_argument('--train-split', metavar='NAME', default='train',
    #                     help='dataset train split (default: train)')
    # group.add_argument('--val-split', metavar='NAME', default='validation',
    #                     help='dataset validation split (default: validation)')
    # group.add_argument('--dataset-download', action='store_true', default=False,
    #                     help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    # group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
    #                     help='path to class to idx mapping file (default: "")')
    # parser.add_argument('--tag', default='ckp', type=str, metavar='TAG')
    # # Model parameters
    # group = parser.add_argument_group('Model parameters')
    # group.add_argument('--model', default='mamba_vision_T', type=str, metavar='MODEL',
    #                     help='Name of model to train (default: "gc_vit_tiny"')
    # group.add_argument('--pretrained', action='store_true', default=False,
    #                     help='Start with pretrained version of specified network (if avail)')
    # group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
    #                     help='Initialize model from this checkpoint (default: none)')
    # group.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='Resume full model and optimizer state from checkpoint (default: none)')
    # # group.add_argument('--loadcheckpoint', default='/scratch/ll5484/ntire25/mambavision/output/test/checkpoint-100.pth.tar', type=str, metavar='PATH',
    # #                     help='Resume full model and optimizer state from checkpoint (default: none)')
    # group.add_argument('--no-resume-opt', action='store_true', default=False,
    #                     help='prevent resume of optimizer state when resuming model')
    # group.add_argument('--num-classes', type=int, default=None, metavar='N',
    #                     help='number of label classes (Model default if None)')
    # group.add_argument('--gp', default=None, type=str, metavar='POOL',
    #                     help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    # group.add_argument('--img_size', type=int, default=224, metavar='N',
    #                     help='Image patch size (default: None => model default)')
    # group.add_argument('--valperiod', type=int, default=1, metavar='N',
    #                     help='epoch / val')
    # group.add_argument('--input_size', default=[3,224,224], nargs=3, type=int,
    #                     metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    # group.add_argument('--crop-pct', default=0.875, type=float,
    #                     metavar='N', help='Input image center crop percent (for validation only)')
    # group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
    #                     help='Override mean pixel value of dataset')
    # group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
    #                     help='Override std deviation of dataset')
    # group.add_argument('--interpolation', default='', type=str, metavar='NAME',
    #                     help='Image resize interpolation type (overrides model)')
    # group.add_argument('-b', '--batch-size', type=int, default=2, metavar='N',
    #                     help='Input batch size for training (default: 128)')
    # group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
    #                     help='Validation batch size override (default: None)')
    # group.add_argument('--channels-last', action='store_true', default=False,
    #                     help='Use channels_last memory layout')
    # scripting_group = group.add_mutually_exclusive_group()
    # scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
    #                     help='torch.jit.script the full model')
    # scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
    #                     help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
    # group.add_argument('--fuser', default='', type=str,
    #                     help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    # group.add_argument('--grad-checkpointing', action='store_true', default=False,
    #                     help='Enable gradient checkpointing through model blocks/stages')

    # # Optimizer parameters
    # group = parser.add_argument_group('Optimizer parameters')
    # group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "sgd"')
    # group.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8, use opt default)')
    # group.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    # group.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='Optimizer momentum (default: 0.9)')
    # group.add_argument('--weight-decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')
    # group.add_argument('--clip-grad', type=float, default=5.0, metavar='NORM',
    #                     help='Clip gradient norm (default: 5.0, no clipping)')
    # group.add_argument('--clip-mode', type=str, default='norm',
    #                     help='Gradient clipping mode. One of ("norm", "value", "agc")')
    # group.add_argument('--layer-decay', type=float, default=None,
    #                     help='layer-wise learning rate decay (default: None)')

    # # Learning rate schedule parameters
    # group = parser.add_argument_group('Learning rate schedule parameters')
    # group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "step"')
    # parser.add_argument('--lr-ep', action='store_true', default=False,
    #                         help='using the epoch-based scheduler')
    # group.add_argument('--lr', type=float, default=1e-5, metavar='LR',
    #                     help='learning rate (default: 1e-3)')
    # group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    # group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
    #                     help='learning rate cycle len multiplier (default: 1.0)')
    # group.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='MULT',
    #                     help='amount to decay each learning rate cycle (default: 0.5)')
    # group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
    #                     help='learning rate cycle limit, cycles enabled if > 1')
    # group.add_argument('--lr-k-decay', type=float, default=1.0,
    #                     help='learning rate k-decay for cosine/poly (default: 1.0)')
    # group.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6)')
    # group.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
    # group.add_argument('--epochs', type=int, default=800, metavar='N',
    #                     help='number of epochs to train (default: 310)')
    # group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
    #                     help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    # group.add_argument('--start-epoch', default=None, type=int, metavar='N',
    #                     help='manual epoch number (useful on restarts)')
    # group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
    #                     help='list of decay epoch indices for multistep lr. must be increasing')
    # group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
    #                     help='epoch interval to decay LR')
    # group.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    # group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    # group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')

    # # Augmentation & regularization parameters
    # group = parser.add_argument_group('Augmentation and regularization parameters')
    # group.add_argument('--no-aug', action='store_true', default=False,
    #                     help='Disable all training augmentation, override other train aug args')
    # group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
    #                     help='Random resize scale (default: 0.08 1.0)')
    # group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
    #                     help='Random resize aspect ratio (default: 0.75 1.33)')
    # group.add_argument('--hflip', type=float, default=0.5,
    #                     help='Horizontal flip training aug probability')
    # group.add_argument('--vflip', type=float, default=0.,
    #                     help='Vertical flip training aug probability')
    # group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
    #                     help='Color jitter factor (default: 0.4)')
    # group.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
    #                     help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    # group.add_argument('--aug-repeats', type=float, default=0,
    #                     help='Number of augmentation repetitions (distributed training only) (default: 0)')
    # group.add_argument('--aug-splits', type=int, default=0,
    #                     help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    # group.add_argument('--jsd-loss', action='store_true', default=False,
    #                     help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    # group.add_argument('--bce-loss', action='store_true', default=False,
    #                     help='Enable BCE loss w/ Mixup/CutMix use.')
    # group.add_argument('--bce-target-thresh', type=float, default=None,
    #                     help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    # group.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
    #                     help='Random erase prob (default: 0.25)')
    # group.add_argument('--remode', type=str, default='pixel',
    #                     help='Random erase mode (default: "pixel")')
    # group.add_argument('--recount', type=int, default=1,
    #                     help='Random erase count (default: 1)')
    # group.add_argument('--resplit', action='store_true', default=False,
    #                     help='Do not random erase first (clean) augmentation split')
    # group.add_argument('--mixup', type=float, default=0.8,
    #                     help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    # group.add_argument('--cutmix', type=float, default=1.0,
    #                     help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    # group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
    #                     help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    # group.add_argument('--mixup-prob', type=float, default=1.0,
    #                     help='Probability of performing mixup or cutmix when either/both is enabled')
    # group.add_argument('--mixup-switch-prob', type=float, default=0.5,
    #                     help='Probability of switching to cutmix when both mixup and cutmix enabled')
    # group.add_argument('--mixup-mode', type=str, default='batch',
    #                     help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
    #                     help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    # group.add_argument('--smoothing', type=float, default=0.1,
    #                     help='Label smoothing (default: 0.1)')
    # group.add_argument('--train-interpolation', type=str, default='random',
    #                     help='Training interpolation (random, bilinear, bicubic default: "random")')
    # group.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
    #                     help='Dropout rate (default: 0.)')
    # group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
    #                     help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    # group.add_argument('--drop-path', type=float, default=0.38, metavar='PCT',
    #                     help='Drop path rate (default: None)')
    # group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
    #                     help='Drop block rate (default: None)')
    # group.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
    #                     help='Drop of the attention, gaussian std')

    # # Batch norm parameters (only works with gen_efficientnet based models currently)
    # group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
    # group.add_argument('--bn-momentum', type=float, default=None,
    #                     help='BatchNorm momentum override (if not None)')
    # group.add_argument('--bn-eps', type=float, default=None,
    #                     help='BatchNorm epsilon override (if not None)')
    # group.add_argument('--sync-bn', action='store_true',
    #                     help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    # group.add_argument('--dist-bn', type=str, default='reduce',
    #                     help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    # group.add_argument('--split-bn', action='store_true',
    #                     help='Enable separate BN layers per augmentation split.')

    # # Model Exponential Moving Average
    # group = parser.add_argument_group('Model exponential moving average parameters')
    # group.add_argument('--model-ema', action='store_true', default=True,
    #                     help='Enable tracking moving average of model weights')
    # group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
    #                     help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    # group.add_argument('--model-ema-decay', type=float, default=0.1998,
    #                     help='decay factor for model weights moving average (default: 0.9998)')

    # # Misc
    # group = parser.add_argument_group('Miscellaneous parameters')
    # group.add_argument('--seed', type=int, default=42, metavar='S',
    #                     help='random seed (default: 42)')
    # group.add_argument('--worker-seeding', type=str, default='all',
    #                     help='worker seed mode (default: all)')
    # group.add_argument('--log-interval', type=int, default=50, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
    #                     help='how many batches to wait before writing recovery checkpoint')
    # group.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
    #                     help='number of checkpoints to keep (default: 3)')
    # group.add_argument('-j', '--workers', type=int, default=8, metavar='N',
    #                     help='how many training processes to use (default: 8)')
    # group.add_argument('--save-images', action='store_true', default=False,
    #                     help='save images of input bathes every log interval for debugging')
    # group.add_argument('--amp', action='store_true', default=False,
    #                     help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    # group.add_argument('--apex-amp', action='store_true', default=False,
    #                     help='Use NVIDIA Apex AMP mixed precision')
    # group.add_argument('--native-amp', action='store_true', default=True,
    #                     help='Use Native Torch AMP mixed precision')
    # group.add_argument('--no-ddp-bb', action='store_true', default=False,
    #                     help='Force broadcast buffers for native DDP to off.')
    # group.add_argument('--pin-mem', action='store_true', default=False,
    #                     help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # group.add_argument('--no-prefetcher', action='store_true', default=True,
    #                     help='disable fast prefetcher')
    # group.add_argument('--output', default='output/test', type=str, metavar='PATH',
    #                     help='path to output folder (default: none, current dir)')
    # group.add_argument('--experiment', default='', type=str, metavar='NAME',
    #                     help='name of train experiment, name of sub-folder for output')
    # group.add_argument('--log_dir', default='./log_dir/', type=str,
    #                     help='where to store tensorboard')
    # group.add_argument('--eval-metric', default='psnr', type=str, metavar='EVAL_METRIC',
    #                     help='Best metric (default: "psnr"')
    # group.add_argument('--tta', type=int, default=0, metavar='N',
    #                     help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    # group.add_argument("--local_rank", default=0, type=int)
    # group.add_argument("--data_len", default=1281167, type=int,help='size of the dataset')

    # group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
    #                     help='use the multi-epochs-loader to save time at the beginning of every epoch')
    # group.add_argument('--log-wandb', action='store_true', default=False,
    #                     help='log training and validation metrics to wandb')
    # group.add_argument('--validate_only', action='store_true', default=False,
    #                     help='run model validation only')

    # group.add_argument('--no_saver', action='store_true', default=False,
    #                     help='Save checkpoints')
    # group.add_argument('--ampere_sparsity', action='store_true', default=False,
    #                     help='Save checkpoints')
    # group.add_argument('--lmdb_dataset', action='store_true', default=False,
    #                     help='use lmdb dataset')
    # group.add_argument('--bfloat', action='store_true', default=False,
    #                     help='use bfloat datatype')
    # group.add_argument('--mesa',  type=float, default=0.25,
    #                     help='use memory efficient sharpness optimization, enabled if >0.0')
    # group.add_argument('--mesa-start-ratio',  type=float, default=0.25,
    #                     help='when to start MESA, ratio to total training time, def 0.25')

    # # GPUs
    # group.add_argument('--gpu_ids', default='0', type=str, metavar='NAME',
    #                     help='available gpus')
    # Parse remaining command-line arguments
    # args = parser.parse_args(remaining_argv)
    # args.dtype = str(torch.float16) 
    
    

    parser.add_argument('--model', default='mamba_vision_T', type=str, metavar='MODEL',
                         help='Name of model to train (default: "gc_vit_tiny"')
    parser.add_argument('--dataset_name', metavar='DIR',type=str, default='DIV2K', help='path to train dataset')
    
    
    # Please change this when tunning
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument('--seed', type=int, default=42, metavar='S',help='random seed (default: 42)')
    parser.add_argument('--use_seed', type=bool, default=False,help='choose to use random seed, type boolean')
    parser.add_argument('--gpu_ids', default='0', type=str, metavar='NAME',help='available gpus')
    parser.add_argument('--WEIGHT_L1', default=0.7, type=float, metavar='N', help='')
    parser.add_argument('--WEIGHT_PEC', default=0.3, type=float, metavar='N', help='')
    
    # you could choose these parameters
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="L2 regularzation coefficient")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument('--bn-momentum', type=float, default=0.9,help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--use_gpu', type=bool, default=True, help="Flag to use GPU if available")
    parser.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--no-prefetcher', action='store_true', default=True,help='disable fast prefetcher')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',help='how many training processes to use (default: 8)')
    parser.add_argument('--pin-mem', action='store_true', default=False,help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Set it directly to data(your FER2013 dataset path)
    parser.add_argument('--data_dir',type=str,default="./dataset")
    parser.add_argument('--json_path',type=str,default="./utils/train_X4.json")
    parser.add_argument('--output_dir',type=str, default="./output")
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',help='torch.jit.script the full model')
    
    # parser.add_argument('--ckp_dir', type=str, default='./ckp', help="Path to checkpoint directory")
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--log_path', type=str, default='./log', help="Path to log file")
    
    # Set it to False When you Run this code first time
    parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--ckp_path', type=str, default= './ckp/VGG_attention_1_epoch/VGG_attention_1_epoch_30_.pth', help="relative path to pretrain ckp")
    
    # Set model parameters
    parser.add_argument('--img_size', type=int, default=224, help="image size for croping and resizing")
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Drop of the attention, gaussian std')
    parser.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.38, metavar='PCT',
                        help='Drop path rate (default: None)')


    args = parser.parse_args(remaining_argv)
    args.dtype = str(torch.float16) 
    # If config file exists, load and override default arguments
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
            args = parser.parse_args(remaining_argv)  # Re-parse with new defaults

    # Save parsed arguments as a YAML string for logging purposes
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    
    return args,args_text

if __name__ == "__main__":
    args,args_text = _parse_args()
    print(args)  # Debugging: print parsed arguments
