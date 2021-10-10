from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim

class SignalTransform:
    """Create two operations of the same signal"""
    def __init__(self, signal_data):
        self.signal_data = signal_data

    def __call__(self):
        adj_factor = np.random.uniform()
        aug_weak = np.random.uniform()
        aug1 = np.random.uniform()
        aug2 = np.random.uniform()
        aug3 = np.random.uniform()
        aug4 = np.random.uniform()
        aug5 = np.random.uniform()

        
        tran_1 = torch.Tensor(transform_signal_1(self.signal_data))
        tran_2 = torch.Tensor(transform_signal_2(self.signal_data))

        return [(tran_1 - torch.min(tran_1)) /
                (torch.max(tran_1) - torch.min(tran_1)),
                (tran_2 - torch.min(tran_2)) /
                (torch.max(tran_2) - torch.min(tran_2))]



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

# def set_optimizer_fine_tuning(opt, backbone, classifier):
def set_optimizer_fine_tuning(backbone):

    # optimizer = optim.SGD(list(backbone.parameters()) + list(classifier.parameters()),
    #                       lr=opt.learning_rate,
    #                       momentum=opt.momentum,
    #                       weight_decay=opt.weight_decay)
    optimizer = optim.SGD(backbone.parameters(),
                          lr=0.0001,
                          momentum=0.9,
                          weight_decay=1e-4)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state



# add the signal transform_1
def transform_signal_1(np_data):

    # Augment_1 TS: Time Swap
    shift = np.random.randint(0, np_data.shape[1] / 2)
    temp = np.copy(np_data[:, 0 : shift + 1])
    np_data[:, 0 : shift + 1] = np_data[:, np_data.shape[1] - shift - 1 : ]
    np_data[:, np_data.shape[1] - shift - 1 : ] = temp

    # Augment_2 AO: Amplitude Offset
    np_data = np_data + np.random.uniform(np.min(np_data), np.max(np_data))

    # Augment_3 FO: Flipping Operation 
    np_data = np.flip(np_data)

    # Augment_4 RCS:  Random Crop Sequence--randomly crop the signal as 3088×1
    rand = np.random.randint(0, 512)
    np_data[:, 0 : rand] = 0
    np_data[:, rand + 3088 :] = 0

    # Augment_5 AWGN: Additive White Gaussian Noise--randomly add 0-1dB Gaussian white noise
    np_data = wgn(np_data, np.random.rand()) + np_data

    return np_data.copy()


# add the signal transform_2
def transform_signal_2(np_data):

    # Augment_1 TS: Time Swap
    shift = np.random.randint(0, np_data.shape[1] / 2)
    temp = np.copy(np_data[:, 0 : shift + 1])
    np_data[:, 0 : shift + 1] = np_data[:, np_data.shape[1] - shift - 1 : ]
    np_data[:, np_data.shape[1] - shift - 1 : ] = temp

    # Augment_2 AO: Amplitude Offset
    np_data = np_data + np.random.uniform(np.min(np_data), np.max(np_data))
  
    # Augment_3 FO: Flipping Operation
    np_data = -np_data

    # Augment_4 RCS:  Random Crop Sequence--randomly crop the signal as 3088×1
    rand = np.random.randint(0, 512)
    np_data[:, 0 : rand] = 0
    np_data[:, rand + 3088 :] = 0

    # Augment_5 AWGN: Additive White Gaussian Noise--randomly add 0-1dB Gaussian white noise
    np_data = wgn(np_data, np.random.rand()) + np_data
        
    return np_data.copy()


# Additive white Gaussian noise
def wgn(x, snr):
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / x.shape[1]
    npower = xpower / snr
    noise_x = np.random.randn(x.shape[1]).reshape(1, x.shape[1], 1)

    return noise_x * np.sqrt(npower)
