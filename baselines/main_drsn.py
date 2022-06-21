from __future__ import print_function
import sys
import argparse
import time
import random
import torch
import numpy as np
import sys
sys.path.append("..")
from main_ce import set_loader
from utils.util import AverageMeter, set_optimizer
from utils.util import accuracy
from networks.DRSN import DRSN_CW





def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    parser.add_argument('--data_folder',
                        type=str,
                        default=None,
                        help='path to custom dataset')
    # model dataset
    parser.add_argument('--model', type=str, default='DRSN')
    parser.add_argument('--dataset', type=str, default='CWRU_signal',
                        choices=['gear_fault_signal', 'CWRU_signal', 'XMU_Motor_signal'], help='dataset')


    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        raise ValueError('No data folder Input!!')

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

   

    return opt


def set_model(opt):
    model = DRSN_CW()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (singals, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        singals = singals.float().cuda(non_blocking=True)
        labels = labels.long().cuda(non_blocking=True)   
        bsz = labels.shape[0]

        # compute loss
        output = model(singals)

        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (signals, labels) in enumerate(val_loader):
            signals = signals.float().cuda()
            labels = labels.long().cuda()
            bsz = labels.shape[0]

            # forward
            output = model(signals)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    best_acc = 0
    opt = parse_option()
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    # optimizer = torch.optim.Adam(model.parameters(),
    #                       lr=opt.learning_rate,
    #                       weight_decay=opt.weight_decay)
    optimizer = set_optimizer(opt, model)
    # lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[40, 80], gamma = 0.1)
    # training routine
    for epoch in range(1, opt.epochs + 1):

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        lr_scheduler.step()
        loss, val_acc = validate(val_loader, model, criterion, opt)

        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
