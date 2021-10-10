import torch
import os
import argparse
from networks.sigsiam_DRSN import sigsiam_DRSN_CW
from datasets.XMU_Motor_dataset import NoTransformMotorSignalDataset

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
            correct_k = correct[:k].contiguous().view(-1).float().sum(
                0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='SigSiam Evaluating!')
    parser.add_argument('--model_path', default=None, type=str, help='model_path')
    parser.add_argument('--train_folder', default='D:/Ftp_Server/zgx/data/CWRU_data/class_balance/train/', type=str, help='train_folder')
    parser.add_argument('--results_txt', default=None, type=str, help='results_txt')

    args = parser.parse_args()        

    model_path = args.model_path
    train_folder = args.train_folder

    model = sigsiam_DRSN_CW()
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model']
    model.cuda()
    model.load_state_dict(state_dict)

    train_dataset = NoTransformMotorSignalDataset(train_folder)

   
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=512,
                                            shuffle=False,
                                            num_workers=8,
                                            pin_memory=True)

    model.eval()
    top1 = AverageMeter()
    # acc1 = 0
    label_list, feature_list = [], []
    with torch.no_grad():
        for batch_idx, (signals, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                signals = signals.float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            bsz = target.shape[0]
            features = model(signals)[2]
            print(features.shape)
            feature_list += features.tolist()
            label_list += target.tolist()
            if batch_idx == 1005:
                break

    if not os.path.exists('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/'):
        os.makedirs('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/')
    with open('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/' + args.results_txt.split('_')[-1] +'.txt', 'w+') as f:
        for i in range(len(label_list)):
            data = str(label_list[i]) + " " + " ".join('%s' % num
                                                    for num in feature_list[i])
            f.write(data)
            f.write('\n')
