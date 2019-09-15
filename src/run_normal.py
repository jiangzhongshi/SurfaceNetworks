#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
import shutil

parser = argparse.ArgumentParser(description='Training Controller, For normal')
parser.add_argument('--model', default='lap')
parser.add_argument('--layer', default='15')
parser.add_argument('--patch', default='1024')
parser.add_argument('--size', default='50k')
parser.add_argument('--num-updates', default='1250')
parser.add_argument('--num-epoch', default=300, type=int)
parser.add_argument('--lr', default='1e-3')
parser.add_argument('--gpu-slot', default=0, type=int)
parser.add_argument('--batch-size', default='1')
parser.add_argument('--additional-opt', default='')
parser.add_argument('--deser', default='')
parser.add_argument('--test-dump', default='')

args = parser.parse_args()
if args.test_dump != '':
    des_opt = (os.path.basename(args.deser).split('_'))
    args.size = des_opt[0] +'/'+des_opt[1]
    args.patch = des_opt[2]
    args.model = des_opt[4]
res = f'{args.size.replace("/","_")}_{args.patch}_{args.lr}_{args.model}'
cmd = f'CUDA_VISIBLE_DEVICES={args.gpu_slot} python normal_predict/train_4_normal.py --input V --output normal --model {args.model} --lay {args.layer} --dat ~/data/Normals/{args.size}/train/{args.patch}/ --test ~/data/Normals/{args.size}/test/{args.patch}/ --plot 1 --batch 32 --num-u {args.num_updates} --lr {args.lr} --res {res} --uni --pre-load --num-e {args.num_epoch} --half-lr 20  --additional-opt {args.additional_opt}'
if args.test_dump != '':
    cmd += ' --debug --only-forward-test --dump-dir '+args.test_dump
else:
    cmd += ' --no-test'
if args.deser != '':
    cmd += f' --deser {args.deser}'
print(cmd)
subprocess.run(cmd, shell=True)
