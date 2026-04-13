import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from model.ResNet_models import Pred_endecoder
from data import get_loader, get_val_loader
from utils import visualize_pred, adjust_lr,visualize_original_img

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--show_iters', type=int, default=100, help='BNN sampling iterations')
parser.add_argument('--data_root', type=str, default='/home/peipei/rgbd_sod/data/', help='data path')

opt = parser.parse_args()
f = open('log.txt', 'a+')
print('========training settings===========')
for k in list(vars(opt).keys()):
    print('%s: %s'%(k, vars(opt)[k]))
    f.writelines('%s: %s\n'%(k, vars(opt)[k]))
# generator
generator = Pred_endecoder(channel=opt.feat_channel)
generator = torch.nn.DataParallel(generator)
# generator.load_state_dict(torch.load('./models/M27_30.pth'))
generator.cuda()
generator_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), opt.lr_gen)


image_root = opt.data_root+'rgbd_old_train_data/RGB/'
gt_root = opt.data_root+'rgbd_old_train_data/GT/'
depth_root = opt.data_root+'rgbd_old_train_data/depth/'
train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

image_root = opt.data_root+'rgbd_old_test_data/DES/RGB/'
gt_root = opt.data_root+'rgbd_old_test_data/DES/GT/'
depth_root = opt.data_root+'rgbd_old_test_data/DES/depth/'
val_loader = get_val_loader(image_root, gt_root, depth_root, batchsize=1, trainsize=opt.trainsize)


size_rates = [1]  # multi-scale training

def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

print("Let's go!")
save_path = 'models/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
losses = [0, 0, 0, 0]
for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()

            images, gts, depths = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            depths = Variable(depths).cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                lefts = F.upsample(lefts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # depths = 1-depths
            att, pred = generator(images, depths)

            loss_rgb = structure_loss(att, gts) + structure_loss(pred, gts)
            loss_all = loss_rgb
            loss_all.backward()
            generator_optimizer.step()

        losses[1] += 0
        losses[0] += loss_rgb.data
        losses[2] += 0
        losses[3] += 0
    
        if i % opt.show_iters == 0:
            msg = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], 1 Loss: {:.4f}, 2 loss: {:.4f}, 3 loss: {:.4f}, 4 loss: {:.4f}'. \
                format(datetime.now(), epoch, opt.epoch, i, total_step, losses[0] / opt.show_iters,
                       losses[1] / opt.show_iters, losses[3] / opt.show_iters, losses[2] / opt.show_iters)
            print(msg)
            f.writelines(msg + '\n')

            visualize_original_img(images)
            visualize_pred(gts, 'gt')
            visualize_pred(depths, 'depths')
            visualize_pred(torch.sigmoid(pred), 'pred')
            visualize_pred(torch.sigmoid(att), 'att')

            #torch.save(generator.state_dict(), save_path + 'Model_newest.pth')
            losses = [0, 0, 0, 0]
        elif i == total_step:
            losses = [0, 0, 0, 0]

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    ### validation dataset
    generator.eval()
    mae_rgb = 0
    for i, pack in enumerate(val_loader, start=1):
        images, gts, depths = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        depths = Variable(depths).cuda()

        att, pred = generator(images, depths)
        mae_rgb = mae_rgb + torch.abs(gts - pred.sigmoid()).mean().data

    msg = 'Validation DES: mae_rgb = %0.4f' % (mae_rgb / len(val_loader))
    print(msg)
    f.writelines(msg + '\n')
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '.pth')
