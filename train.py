import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet20
import resnet50
from utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','ImageNet'])
parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20','resnet50'])
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset',default="/home/nie/f/dataset/cifar10/")
parser.add_argument('--pretrain', type=str,
                    default="output/epoch_186_loss_1.269.pth",
                    # default="",
                    help='path to pretrained model')
parser.add_argument('--model_dir', type=str,
                    help='path to save model',default="output")

args = parser.parse_args()

def main():
    if args.model == 'resnet20':
        # model = models.resnet18(pretrained=False)
        model = resnet20.resnet20()
    else:
        model = resnet50.resnet50()
    
    if args.pretrain:
        state_dict = torch.load(args.pretrain)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    
    if (args.gpu is not None):
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[80, 120, 160, 180], last_epoch=-1)

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010]),
            ]), download=True)
        val_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010]),
            ]), download=True)
    else:
        train_dataset = datasets.ImageFolder(
            root=args.data_dir, 
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ]))
        val_dataset = datasets.ImageFolder(
            root=args.data_dir, 
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ]))        

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    losses = AverageMeter('Loss', ':2.4f')
    top1_train = AverageMeter('Acc@1', ':6.2f')
    top5_train = AverageMeter('Acc@5', ':6.2f')
    top1_val = AverageMeter('Acc@1', ':6.2f')
    top5_val = AverageMeter('Acc@5', ':6.2f')

    for epoch in range(args.epochs):
        losses.reset()
        top1_train.reset()
        top5_train.reset()
        top1_val.reset()
        top5_val.reset()

        model.train()

        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1_train.update(acc1[0], input.size(0))
            top5_train.update(acc5[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch:[{}][{:5d}/{}]\t{}\t{}\t{}".format(epoch, i, len(train_loader), losses, top1_train, top5_train))
        
        model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                output = model(input)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1_val.update(acc1[0], input.size(0))
                top5_val.update(acc5[0], input.size(0))
                
            print("Valid:\t{}\t{}\n".format(top1_train, top5_train))

        lr_scheduler.step()

        torch.save(model.state_dict(), os.path.join(args.model_dir, "epoch_{}_loss_{:.3f}.pth".format(epoch, losses.avg)))


if __name__ == '__main__':
    main()