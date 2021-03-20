import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import utils
from dataset import VOCDataset
from search.architect_coop import Architect, softXEnt
from search.model_search import Network
from yoloLoss import yoloLoss


def get_args():
    parser = argparse.ArgumentParser("yoloSGL")

    # file io
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')

    # hyper parameter config
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--weight_lambda', type=float, default=1.0)

    # training config
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--is_parallel', type=int, default=0)
    parser.add_argument('--debug', default=False, action='store_true')

    # network config
    parser.add_argument('--init_channels', type=int, default=3, help='num of init channels')
    parser.add_argument('--layers', type=int, default=6, help='total number of layers')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')

    args = parser.parse_args()
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.unrolled = False

    return args


def main():
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)

    # setup log and random seed
    args = get_args()
    logging.info("args = %s", args)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    # fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # models
    criterion = yoloLoss(7, 2, 5, 0.5)
    criterion = criterion.cuda()

    model1 = Network(args.init_channels, args.layers, criterion).cuda()
    model2 = Network(args.init_channels, args.layers, criterion).cuda()
    logging.info("param size of model1 = %fMB", utils.count_parameters_in_MB(model1))
    logging.info("param size of model2 = %fMB", utils.count_parameters_in_MB(model2))

    optimizer1 = torch.optim.SGD(
        model1.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    optimizer2 = torch.optim.SGD(
        model2.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model1, model2, args)

    # prepare data
    data = VOCDataset("trainval")
    num_samples = len(data)
    lengths = [num_samples // 3, num_samples // 3, num_samples - 2 * (num_samples // 3)]
    train_data, valid_data, external_data = torch.utils.data.random_split(data, lengths, generator=torch.Generator().manual_seed(args.seed))
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=False)
    external_queue = torch.utils.data.DataLoader(external_data, batch_size=args.batch_size, pin_memory=False)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, pin_memory=False)

    # main loop
    for epoch in range(args.epochs):
        # log current progress
        lr1, lr2 = scheduler1.get_last_lr()[0], scheduler2.get_last_lr()[0]
        logging.info('epoch %d lr1 %e lr2 %e', epoch, lr1, lr2)

        genotype1 = model1.genotype()
        genotype2 = model2.genotype()
        logging.info('genotype1 = %s', genotype1)
        logging.info('genotype2 = %s', genotype2)

        # training
        train_acc1, train_obj1, train_acc2, train_obj2 = train(
            args,
            train_queue,
            valid_queue,
            external_queue,
            model1,
            model2,
            architect,
            criterion,
            optimizer1,
            optimizer2,
            lr1,
            lr2)
        logging.info('train_acc1 %f train_acc2 %f', train_acc1, train_acc2)

        # validation
        valid_acc1, valid_obj1, valid_acc2, valid_obj2 = infer(
            args,
            valid_queue,
            model1,
            model2,
            criterion)
        logging.info('valid_acc1 %f valid_acc2 %f', valid_acc1, valid_acc2)

        # save and next
        utils.save(model1, os.path.join(args.save, 'weights1.pt'))
        utils.save(model2, os.path.join(args.save, 'weights2.pt'))
        scheduler1.step()
        scheduler2.step()


def train(args,
          train_queue,
          valid_queue,
          external_queue,
          model1,
          model2,
          architect,
          criterion,
          optimizer1,
          optimizer2,
          lr1,
          lr2):
    objs_1 = utils.AvgrageMeter()
    top1_1 = utils.AvgrageMeter()
    top5_1 = utils.AvgrageMeter()

    objs_2 = utils.AvgrageMeter()
    top1_2 = utils.AvgrageMeter()
    top5_2 = utils.AvgrageMeter()

    valid_queue_iter = iter(valid_queue)
    external_queue_iter = iter(external_queue)
    for step, (input, target) in enumerate(train_queue):
        model1.train()
        model2.train()
        n = input.size(0)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        try:
            input_external, target_external = next(external_queue_iter)
        except:
            external_queue_iter = iter(external_queue)
            input_external, target_external = next(external_queue_iter)

        target = target.cuda()
        target_search = target_search.cuda()
        target_external = target_external.cuda()
        input = input.cuda()
        input_search = input_search.cuda()
        input_external = input_external.cuda()

        # import ipdb; ipdb.set_trace()
        architect.step(input, target,
                       input_external, target_external,
                       input_search, target_search,
                       lr1, lr2, optimizer1, optimizer2, unrolled=args.unrolled)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        logits1 = model1(input)
        logits2 = model2(input)

        loss1 = criterion(logits1, target)
        loss2 = criterion(logits2, target)

        external_out1 = model1(input_external)
        external_out2 = model2(input_external)
        softlabel_other1 = F.softmax(external_out1, 1)
        softlabel_other2 = F.softmax(external_out2, 1)
        loss_soft1 = softXEnt(external_out2, softlabel_other1)
        loss_soft2 = softXEnt(external_out1, softlabel_other2)

        loss_all = loss1 + loss2 + args.weight_lambda * (loss_soft2 + loss_soft1)

        loss_all.backward()

        nn.utils.clip_grad_norm_(model1.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(model2.parameters(), args.grad_clip)
        optimizer1.step()
        optimizer2.step()

        prec1, prec5 = utils.accuracy(logits1, target, topk=(1, 5))
        objs_1.update(loss1.item(), n)
        top1_1.update(prec1.item(), n)
        top5_1.update(prec5.item(), n)

        prec1, prec5 = utils.accuracy(logits2, target, topk=(1, 5))
        objs_2.update(loss2.item(), n)
        top1_2.update(prec1.item(), n)
        top5_2.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train 1st %03d %e %f %f', step, objs_1.avg, top1_1.avg, top5_1.avg)
            logging.info('train 2nd %03d %e %f %f', step, objs_2.avg, top1_2.avg, top5_2.avg)

    return top1_1.avg, objs_1.avg, top1_2.avg, objs_2.avg


def infer(args, valid_queue, model1, model2, criterion):
    objs_1 = utils.AvgrageMeter()
    top1_1 = utils.AvgrageMeter()
    top5_1 = utils.AvgrageMeter()
    objs_2 = utils.AvgrageMeter()
    top1_2 = utils.AvgrageMeter()
    top5_2 = utils.AvgrageMeter()

    model1.eval()
    model2.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model1(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs_1.update(loss.item(), n)
            top1_1.update(prec1.item(), n)
            top5_1.update(prec5.item(), n)

            # for the second model.
            logits = model2(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs_2.update(loss.item(), n)
            top1_2.update(prec1.item(), n)
            top5_2.update(prec5.item(), n)
            if step % args.report_freq == 0:
                logging.info('valid 1st %03d %e %f %f', step, objs_1.avg, top1_1.avg, top5_1.avg)
                logging.info('valid 2nd %03d %e %f %f', step, objs_2.avg, top1_2.avg, top5_2.avg)

    return top1_1.avg, objs_1.avg, top1_2.avg, objs_2.avg

if __name__ == '__main__':
    main()