"""
p = min E(p,q) + Ent(p) + VAT(p)
q = min E(p,q)
label = argmax q
fintuing on source model
"""
import random
import time
import warnings
import sys
import argparse
import copy
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from dalib.modules.classifier import ImageClassifier
from dalib.adaptation.sfda_ent_vat import DataSetIdx, VirtualAdversarialLoss, entropy_loss, LabelOptimizer, weight_reg_loss
# from dalib.adaptation.sfda_ent_vat_aug import LabelOptimizer
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy
from tools.transforms import train_transform_aug0, train_transform_aug1, train_transform_aug2, train_transform_aug3, train_transform_aug4
from tools.transforms import val_transform
from tools.lr_scheduler import StepwiseLR3


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
    if args.aug_ind == 0:
        train_transform = train_transform_aug0
    elif args.aug_ind == 1:
        train_transform = train_transform_aug1
    elif args.aug_ind == 2:
        train_transform = train_transform_aug2
    elif args.aug_ind == 3:
        train_transform = train_transform_aug3
    elif args.aug_ind == 4:
        train_transform = train_transform_aug4
    
    dataset = datasets.__dict__[args.data]
    # trainset
    trainset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    trainset_with_index = DataSetIdx(trainset)
    train_loader = DataLoader(trainset_with_index, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, drop_last=True)
    # pseudoset
    pseudoset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    pseudoset_with_index = DataSetIdx(pseudoset)
    pseudo_loader = DataLoader(pseudoset_with_index, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, drop_last=False)
    
    pseudoset_aug = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    pseudoset_with_index_aug = DataSetIdx(pseudoset_aug)
    pseudo_loader_aug = DataLoader(pseudoset_with_index_aug, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, drop_last=False)
    # valset
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, evaluate=True, download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    # create source model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=False)
    src_model = ImageClassifier(backbone, trainset.num_classes, weight_norm=args.wn, no_relu=args.no_relu).to(device)

    # load pretrained source model
    print(f"=> loading source model: {args.data}_{args.source}_{args.arch}.pth")
    src_state_dict = torch.load(f'source_models/{args.data}_{args.source}_{args.arch}.pth')
    src_model.load_state_dict(src_state_dict)

    # create target model
    trg_model = copy.deepcopy(src_model)

    # freeze source model
    for param in src_model.parameters():
        param.requires_grad = False
    src_model.eval()

    # freeze target classifer
    # for param in trg_model.head.parameters():
    #     param.requires_grad = False

    # define optimizer and lr scheduler
    optimizer = SGD(trg_model.get_parameters(), args.lr, momentum=args.momentum, 
                    weight_decay=args.weight_decay, nesterov=True)
    # optimizer = Adam(trg_model.get_parameters(), args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = StepwiseLR1(optimizer, max_iter=args.epochs*len(train_loader), init_lr=args.lr)
    # lr_scheduler = StepwiseLR2(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler = StepwiseLR3(optimizer, init_lr=args.lr)

    # define label optimizer
    label_optim = LabelOptimizer(N=len(trainset), K=trainset.num_classes, lambd=args.lambd, device=device)
    virt_adv_loss = VirtualAdversarialLoss()
    
    # start training
    print("=> start training")
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, pseudo_loader, pseudo_loader_aug, src_model, trg_model, label_optim, 
                virt_adv_loss, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        if args.data == "VisDA2017":
            acc1 = validate_per_class(val_loader, trg_model, args)
        else:
            acc1 = validate(val_loader, trg_model, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(trg_model.state_dict())
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    trg_model.load_state_dict(best_model)
    if args.data == "VisDA2017":
        acc1 = validate_per_class(test_loader, trg_model, args)
    else:
        acc1 = validate(test_loader, trg_model, args)
    print("test_acc1 = {:3.1f}".format(acc1))


def train(train_loader: DataLoader, pseudo_loader: DataLoader, pseudo_loader_aug: DataLoader,
          src_model: ImageClassifier, trg_model: ImageClassifier, label_optim: LabelOptimizer, 
          virt_adv_loss: VirtualAdversarialLoss, optimizer: SGD, 
          lr_scheduler: StepwiseLR3, epoch: int, args: argparse.Namespace):
    
    batch_time = AverageMeter('Time', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # (1) optimize label
    print(f"=> Epoch: [{epoch}] optimizing labels")
    start = time.time()
    trg_model.eval()
    with torch.no_grad():
        # update matrix P, no aug
        P1 = torch.zeros_like(label_optim.P).to(device)
        for _, (images, _, index) in enumerate(pseudo_loader):
            # compute output
            images = images.to(device)
            y_t, _ = trg_model(images)
            p_t = F.softmax(y_t, dim=1)
            label_optim.update_P(p_t, index)
        P1 += label_optim.P.clone()

        # aug
        P2 = torch.zeros_like(label_optim.P).to(device)
        for i in range(args.num_aug):
            for _, (images, _, index) in enumerate(pseudo_loader_aug):
                # compute output
                images = images.to(device)
                y_t, _ = trg_model(images)
                p_t = F.softmax(y_t, dim=1)
                label_optim.update_P(p_t, index)
            P2 += label_optim.P.clone()
        P2 /= args.num_aug

        label_optim.P = P1 * args.aug_coeff + P2 * (1 - args.aug_coeff)
        label_optim.update_Labels()
    
    # check label acc
    correct = 0
    with torch.no_grad():
        for _, (images, labels, index) in enumerate(pseudo_loader):
            labels = labels.to(device)
            pseudo_labels = label_optim.Labels[index]
            correct += pseudo_labels.eq(labels).sum().item()
    acc = correct / len(train_loader.dataset)

    print(f"pseudo label acc: {acc}, optimizing label time: {time.time() - start}")

    # (2) optimize model
    end = time.time()
    trg_model.train()
    for i, (images, _, index) in enumerate(train_loader):
        lr_scheduler.step()
        images = images.to(device)
        pseudo_labels = label_optim.Labels[index]

        # compute output
        y_t, _ = trg_model(images)
        p_t = F.softmax(y_t, dim=1)

        cls_loss = F.cross_entropy(y_t, pseudo_labels)
        ent_loss = entropy_loss(p_t)
        vat_loss = virt_adv_loss(trg_model, images)

        if args.wr_model == "cls":
            wr_loss = weight_reg_loss(src_model.head, trg_model.head)
        elif args.wr_model == "model":
            wr_loss = weight_reg_loss(src_model, trg_model)
        elif args.wr_model == "none":
            wr_loss = torch.tensor(0.0).cuda()
        
        loss = cls_loss * args.cls_param + (ent_loss + vat_loss) * args.ent_param + wr_loss * args.wr_param

        losses.update(loss.item(), images.size(0))

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
    losses = AverageMeter('Loss', ':.4e')
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
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--wn', action='store_true', default=False,
                        help='use weight norm')
    parser.add_argument('--no_relu', action='store_true', default=False,
                        help='delete relu in bottleneck')
    parser.add_argument('--lambd', default=0.5, type=float,
                        help='the trade-off hyper-parameter for optimizing labels')
    parser.add_argument('--cls_param', default=1., type=float,
                        help='the trade-off hyper-parameter for ce loss')
    parser.add_argument('--ent_param', default=1., type=float,
                        help='the trade-off hyper-parameter for entropy loss')
    parser.add_argument('--num_aug', default=4, type=int, metavar='N',
                        help='number of augmentations')
    parser.add_argument('--aug_coeff', default=0.5, type=float,
                        help='the trade-off hyper-parameter for non-aug')
    parser.add_argument('--wr_model', default='model', help='cls or model or none')
    parser.add_argument('--wr_param', default=0.1, type=float,
                        help='the trade-off hyper-parameter for weight regularization loss')
    parser.add_argument('--aug_ind', default=1, type=int)
    
    args = parser.parse_args()
    print(args)
    main(args)

