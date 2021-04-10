import random
import time
import warnings
import sys
import argparse
import copy
import numpy as np
import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import torch.utils.data.distributed
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from dalib.modules.classifier import ImageClassifier
from dalib.adaptation.train_source import cross_entropy_ls, DatasetTransform
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy
from tools.transforms import train_transform, val_transform
from tools.lr_scheduler import StepwiseLR


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code

    dataset = datasets.__dict__[args.data]
    # source
    src_dataset = dataset(root=args.root, task=args.source, download=True)
    len_train = int(0.9 * len(src_dataset))
    len_val = len(src_dataset) - len_train
    trainset_i, valset_i = random_split(src_dataset, [len_train, len_val])
    trainset = DatasetTransform(trainset_i, train_transform)
    valset = DatasetTransform(valset_i, val_transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, drop_last=False)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = src_dataset.num_classes
    classifier = ImageClassifier(backbone, num_classes, weight_norm=args.wn, no_relu=args.no_relu).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_sheduler = StepwiseLR(optimizer, max_iter=args.epochs*len(train_loader), init_lr=args.lr)

    # start training
    best_acc1 = 0.
    for epoch in range(1, args.epochs+1):
        print(lr_sheduler.get_lr())
        # train for one epoch
        train(train_loader, classifier, optimizer, lr_sheduler, epoch, args)

        # evaluate on validation set
        if args.data == "VisDA2017":
            acc1 = validate_per_class(val_loader, classifier, args)
        else:
            acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # save source model
    model_dir = 'source_models'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    save_name = f'{model_dir}/{args.data}_{args.source}_{args.arch}.pth'
    torch.save(best_model, save_name)


def train(train_loader: DataLoader, model: ImageClassifier, optimizer: SGD,
          lr_sheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if lr_sheduler is not None:
            lr_sheduler.step()

        images = images.to(device)
        target = target.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, _ = model(images)

        cls_loss = F.cross_entropy(y_s, target)
        loss = cls_loss

        cls_acc = accuracy(y_s, target)[0]

        losses.update(loss.item(), images.size(0))
        cls_accs.update(cls_acc.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate_per_class(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time,],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    predicts = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            predict = output.argmax(dim=1)
            
            predicts.append(predict)
            targets.append(target)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        predicts = torch.cat(predicts)
        targets = torch.cat(targets)
        
        matrix = confusion_matrix(targets.cpu().float(), predicts.cpu().float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)

        print(f' * Acc@1: {aacc}, Accs: {acc}')

    return aacc


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--wn', action='store_true', default=False,
                        help='use weight norm')
    parser.add_argument('--no_relu', action='store_true', default=False,
                        help='delete relu in bottleneck')
    
    args = parser.parse_args()
    print(args)
    main(args)

