import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings


def run_training(train_module, train_name):
    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = True

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('train_module', type=str)
    parser.add_argument('train_name', type=str)
    # parser.add_argument('--debug', type=int, default=0)
    # parser.add_argument('--threads', type=int, default=0)

    args = parser.parse_args()

    run_training(args.train_module, args.train_name)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
