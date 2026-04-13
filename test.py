import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from model.ResNet_models import Pred_endecoder
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from tqdm import tqdm
from model.functions import RefUnet



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

dataset_path = '/home/peipei/research/rgbd_sod/data/rgbd_test_data/'

generator = Pred_endecoder(channel=opt.feat_channel)
generator = torch.nn.DataParallel(generator)
generator.load_state_dict(torch.load('./models/Model_100.pth'))
generator.cuda()
generator.eval()


test_datasets = ['NJUD','STERE', 'RGBD135', 'NLPR', 'LFSD', 'SIP']

def inv_sigmoid(x):
    y = torch.log((x+1e-10)/(1-x+1e-10))
    return y

for dataset in test_datasets:
    save_path1 = './res_tmp/' + dataset + '/'
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)

    image_root = dataset_path + dataset + '/test_images/'
    depth_root = dataset_path+dataset+'/test_depths/'
    test_loader = test_dataset(image_root, depth_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        image, depth, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        depth = depth.cuda()
        rgb_init, rgb_ref = generator(image, depth)

        res = rgb_ref
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path1 + name, res)


