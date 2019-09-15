import torch
import sys
import os
import plotly
import numpy as np
import argparse
import torch.nn.functional as F
import random
import time
import re
import gc
import glob
import tqdm
import functools
import subprocess
import multiprocessing
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import timer_utils
    from utils import utils_pt as util
    from models import DirDeepModel, LapDeepModel, IdDeepModel, AvgModel, MlpModel, LapMATModel, GatDeepModel, EfficientCascade, GlobalLocalModel
    import sampler
else:
    from ..utils import utils_pt as util
    from .models import DirDeepModel, LapDeepModel, IdDeepModel, AvgModel, MlpModel, LapMATModel, GatDeepModel, EfficientCascade, GlobalLocalModel
    from . import sampler

random.seed(17)
# Training settings
parser = argparse.ArgumentParser(description='Normal Predictor')

# Model Setting
parser.add_argument('--input-type', default=[],
                    action='append',
                    choices=[ 'V'],
                   help='VNHK|wks| G')
parser.add_argument('--output-type', default='10class',
                   help='100[vdist|geodist|pndist|lapdist]')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--model', default="lap",
                    help='lap | dirac | avg | mlp | cas | globallocal')
parser.add_argument('--layer', type=int, default=15)

parser.add_argument('--no-cuda', action='store_true', default=False)

# Training Process Control
parser.add_argument('--deser',  default=None,
                    help='deserialization path: ')
parser.add_argument('--result-prefix', default='debug')
parser.add_argument('--data-path', default='/dev/shm/')
parser.add_argument('--test-path', default='@')
parser.add_argument('--num-epoch', type=int, default=500, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--num-updates', type=int, default=500, metavar='N',
                    help='num of training epochs (default: 500)')
parser.add_argument('--no-test', action="store_true")
parser.add_argument('--half-lr', type=int, default=-1,
                    help='Halves lr every N epochs, -1 means no')
parser.add_argument('--only-forward-test', action="store_true", help="Used to generate results")
parser.add_argument('--dump-dir', default='/dev/shm/')

# Optimizing
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--dense', action='store_true')

# Experimental Options
parser.add_argument('--uniform-mesh', action="store_true",
                    help='Scale Mesh to uniform for training across categories')

parser.add_argument(
    '--additional-opt',
    default=[],
    action='append',
    choices=[
        'hack1',
        'intrinsic',
        'amsgrad',
        ''])

parser.add_argument('--debug', action='store_true', help='Not writing to file')
parser.add_argument('--pre-load', action='store_true', help='Offload computation')
parser.add_argument('--plot-interval', type=int, default=30, help='Save plotly plot interval')

def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    vis = None
    result_identifier = args.result_prefix
    if args.plot_interval <= 0:
        import visdom
        vis = visdom.Visdom(port=2002, env=result_identifier)

    def custom_logging(stuff):
        if args.debug:
            print(f'{result_identifier}::{stuff}', file=sys.stderr) # also to err
        else:
            print(f'{result_identifier}::{stuff}') # also to out
            logfile = f'normal_predict/log/{result_identifier}.log'
            with open(logfile,'a') as fp: print(stuff, file=fp)

    custom_logging(args)
    custom_logging(subprocess.check_output('hostname'))
    custom_logging(subprocess.check_output('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True))

    sample_batch_train = lambda seq: sampler.sample_batch(seq, args, is_fixed=True)
    sample_batch_test = lambda seq: sampler.sample_batch(seq, args, is_fixed=False)
    sampler.sample_batch.EPOCH_FLAG = False

    def loss_fun(inputs, mask, targets, **kwargs):
        inputs = F.normalize(inputs, p=2, dim=2)
        inner = torch.sum(inputs * targets, dim=2)
        l = 1-inner**2
        return torch.mean(torch.masked_select(l ,mask.view(l.size(0),l.size(1)).byte()))
    def mean_angle_deviation(inputs, mask, targets, **kwargs):
        inputs = F.normalize(inputs, p=2, dim=2)
        inner = torch.sum(inputs * targets, dim=2)
        inner = torch.clamp(torch.abs(inner),0,1)
        l = torch.acos(inner)
        return torch.mean(torch.masked_select(l ,mask.view(l.size(0),l.size(1)).byte()))


    input_type_to_dim = {'V':3, 'G':1, 'wks':100, 'cor_V':3, 'N':3, 'curv4':4}
    args.input_dim = sum([input_type_to_dim[i] for i in args.input_type])
    args.output_dim = 3

    bnmode = '' # normal bn
    if 'groupnorm' in args.additional_opt:
        bnmode = 'group'
    if 'nobn' in args.additional_opt:
        bnmode = None

    args.bottleneck = False
    lap_opts = {'layers': args.layer,
                'bnmode': bnmode,
                'only_lap': 'only_lap' in args.additional_opt,
                'nofirstId':False}
    if 'avg' in args.model:
        custom_logging("Using AVG")
        model = AvgModel(args.input_dim, args.output_dim, args.layer)
    elif args.model == 'lap':
        model = LapDeepModel(args.input_dim, args.output_dim, **lap_opts)
    elif args.model.startswith('dirac'):
        model = DirDeepModel(in_features=3, out_features=3, layers=args.layer)

    model.to(device)

    custom_logging("Num parameters {}".format(sum(p.numel() for p in model.parameters())))


    if args.optimizer == 'adam':
        early_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad= 'amsgrad' in args.additional_opt)
    elif args.optimizer == 'sgd':
        custom_logging(f'Using SGD with deser {args.deser}')
        early_optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)

    if args.deser:
        custom_logging('Continue...')
        checkpoint = torch.load(args.deser)
        if type(checkpoint) is dict:
            model.fuzzy_load(checkpoint['weights'])
            early_optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        else:
            print('Warning: Optimizer is not loaded')
            model.fuzzy_load(checkpoint)

    sys.stdout.flush()

    if not args.only_forward_test:
        seq_names =  sorted(glob.glob(args.data_path + '**/*.obj', recursive=True))
    else:
        seq_names = []
    custom_logging(f'SEQ:{len(seq_names)}')

    if args.test_path != '@':
        train_seq_names = seq_names
        test_seq_names = sorted(glob.glob(f'{args.test_path}/**/*.obj', recursive=True))
    else:
        # 80/20 seperation
        sep_length = len(seq_names)//10*8
        random.shuffle(seq_names)
        train_seq_names = seq_names[:sep_length]
        test_seq_names = seq_names[sep_length:]

    real_epoch_counter = 0
    sampler.sample_batch.train_id = 0

    if args.pre_load:
        readnpz = functools.partial(sampler.read_npz, args=args) # pickle-able for imap
        torch.multiprocessing.set_sharing_strategy('file_system') # https://github.com/pytorch/pytorch/issues/973
        if not args.only_forward_test:
            if True:
                train_seq_names = [readnpz(t) for t in tqdm.tqdm(train_seq_names, ncols=0)]
            else:
                with torch.multiprocessing.Pool(1) as p:
                    train_seq_names = list(tqdm.tqdm(p.imap(readnpz, train_seq_names),
                                                    total=len(train_seq_names),
                                                    ncols=0))
            train_seq_names = [t for t in train_seq_names if t is not None]


        if not args.no_test:
            test_seq_names = [readnpz(t) for t in tqdm.tqdm(test_seq_names, ncols=0)]
            # with torch.multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            #     test_seq_names  = list(tqdm.tqdm(p.imap(readnpz, test_seq_names),
            #                                       total=len(test_seq_names),
            #                                       ncols=0))
            test_seq_names = [t for t in test_seq_names if t is not None]
        print('Train size:', len(train_seq_names), ' Test size:', len(test_seq_names))

    train_loss = []
    test_loss = []

    sampler.sample_batch.EPOCH_FLAG=True
    for epoch in range(args.start_epoch,args.num_epoch):
        if not  args.only_forward_test:
            if sampler.sample_batch.EPOCH_FLAG:
                random.shuffle(train_seq_names)
                custom_logging('SHUFFLE')
                sampler.sample_batch.EPOCH_FLAG=False

            model.train()
            loss_value = 0
            mad = 0
            # Train
            pb = tqdm.trange(args.num_updates, ncols=0)
            for num_up in pb:
                inputs, targets, mask, DL, _, curr_name, loss_weight = sample_batch_train(train_seq_names)
                outputs = model(DL, mask, inputs)

                if (torch.isnan(outputs.detach())).any(): assert False, f'NANNNN {curr_name[0]} outputs'
                loss = loss_fun(outputs, mask, targets)
                mad += mean_angle_deviation(outputs,mask, targets).item()

                early_optimizer.zero_grad()
                loss.backward()
                early_optimizer.step()
                loss_value += loss.item()
                if np.isnan(loss_value): assert False, f'NANNNN {curr_name[0]} LOSS'
                pb.set_postfix(loss=loss_value/(num_up+1), mad = mad/(num_up+1))

            custom_logging("Train {}, loss {}, mad {}, time {}".format(epoch, loss_value / args.num_updates, mad/args.num_updates, pb.last_print_t - pb.start_t))
            train_loss.append(loss_value / args.num_updates)

        # Evaluate
        with torch.no_grad():
            loss_value = 0
            mad = 0
            test_trials = (int)(np.ceil(len(test_seq_names) / args.batch_size))
            sampler.sample_batch.test_id = 0
            if not args.no_test:
                for _ in tqdm.trange(test_trials, ncols=0):
                    inputs, targets, mask, DL, _, names, loss_weight = sample_batch_test(test_seq_names)

                    outputs = model(DL, mask, inputs)
                    loss = loss_fun(outputs, mask, targets, loss_weight=loss_weight)

                    loss_value += loss.item()
                    mad += mean_angle_deviation(outputs, mask, targets).item()

                    if args.only_forward_test:
                        directory = f'{args.dump_dir}/{args.result_prefix}/'
                        if not os.path.exists(directory): os.makedirs(directory)
                        for name, targ in zip(names, outputs):
                            np.savetxt(directory + os.path.basename(name) + '.csv', targ.cpu().numpy(), delimiter=',')

                custom_logging("Eval {}, loss {}, mad {}".format(epoch, loss_value / test_trials, mad/test_trials))
                test_loss.append(loss_value / test_trials)
            sys.stdout.flush()
        if args.only_forward_test:
            return
        if epoch % 10 == 9 and not args.debug:
            torch.save({'weights':model.state_dict(), 'optimizer':early_optimizer.state_dict(), 'epoch':epoch}, 'normal_predict/pts/' + args.result_prefix + '_normal_state.pts')

        if args.half_lr > 0 and epoch > 100 and epoch % args.half_lr == 0:
            for param_group in early_optimizer.param_groups:
                param_group['lr'] *= 0.5
            custom_logging(f'Halving LR to {param_group["lr"]}')

    torch.save(model.state_dict(), 'normal_predict/pts/' + args.result_prefix + '_normal_state.pts')

if __name__ == '__main__':
    main()
