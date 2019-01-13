from trainer import Trainer
import argparse
from PIL import Image
from re_evaluate_gan import Evaluator
import os

parser = argparse.ArgumentParser()
parser.add_argument("--type", default='gan')
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--diter", default=5, type=int)
parser.add_argument("--cls", default=False, action='store_true')
parser.add_argument("--vis_screen", default='gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--pre_trained_disc', default=None)
parser.add_argument('--pre_trained_gen', default=None)
parser.add_argument('--pre_trained_encod', default=None)
parser.add_argument('--dataset', default='birds')
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--visualize', default=True, type=bool)
args = parser.parse_args()

evaluator = Evaluator(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  vis_screen=args.vis_screen,
                  save_path=args.save_path,
                  pre_trained_disc="checkpoints_gan/disc_",
                  pre_trained_gen="checkpoints_gan/gen_",
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs,
                  visualize=args.visualize,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  )

evaluator.evaluate_gan()
