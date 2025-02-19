from PIL import Image
import numpy as np
from dataset_for_train import Fusion_dataset2
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from model_TII import BiSeNet
from loss import OhemCELoss, Fusionloss
import torch
from torch.utils.data import DataLoader
import warnings
from Net import FusionNet, WO_ViT, WO_MSFF, WO_CMFF, WO_SFI
from utils import RGB2YCrCb, YCbCr2RGB
warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()
def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()






def train_fusion(num=0, logger=None, args=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr_start = 0.001
    model_pth = './model'
    Method = 'Fusion'
    model_pth = osp.join(model_pth, Method)

    fusion_model = eval('WO_CMFF')(output=1)
    fusion_model.to(device)
    fusion_model.train()


    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr_start)

    n_classes = 9
    seg_model = BiSeNet(n_classes=n_classes)
    save_pth = osp.join(model_pth, 'model_final.pth')
    if logger == None:
           logger = logging.getLogger()
           setup_logger(model_pth)

    seg_model.load_state_dict(torch.load(save_pth))
    seg_model.to(device)
    seg_model.eval()
    for p in seg_model.parameters():
        p.requires_grad = False

    print('Load Segmentation Model {} Successfully~'.format(save_pth))

    train_dataset = Fusion_dataset2('MSRS')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)

    score_thres = 0.7
    ignore_idx = 255
    n_min = 8 * 640 * 480 // 8 - 1
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    criteria_fusion = Fusionloss()




    epoch = 4
    st = glob_st = time.time()
    logger.info(f'Training Fusion Model_WO_MSFF start{st}~')
    for epo in range(0, epoch):
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):

            fusion_model.train()
            image_vis = image_vis.to(device)
            image_ir = image_ir.to(device)
            label = label.to(device)


            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(image_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)

            out, mid = seg_model(image_vis)

            logits = fusion_model(vi_Y, image_ir, out)
            fusion_image = YCbCr2RGB(logits, vi_Cb,vi_Cr)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            lb = torch.squeeze(label, 1)

            optimizer.zero_grad()

            loss_fusion, loss_in, loss_grad, loss_ssim = criteria_fusion(
                torch.cat([vi_Y,vi_Cb,vi_Cr],dim=1), image_ir, label, logits, num
            )
            out, mid = seg_model(fusion_image)
            lossp = criteria_p(out, lb)
            loss2 = criteria_16(mid, lb)
            seg_loss = lossp + 0.1 * loss2

            loss_total = loss_fusion + (num) * seg_loss
            loss_total.backward()

            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))

            if now_it % 10 == 0:
                loss_seg=seg_loss.item()
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_seg: {loss_seg:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim.item(),
                    loss_seg=loss_seg,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
            if now_it % 100 == 0:
                fusion_model_file = os.path.join(model_pth, f'fusion_model_WO_CMFF_it{now_it}.pth')
                torch.save(fusion_model.state_dict(), fusion_model_file)
                logger.info("Fusion Model_WO_CMFF after it {} Save to: {}".format(now_it, fusion_model_file))


        # every epoch save the fusion model
        fusion_model_file = os.path.join(model_pth, f'fusion_model_epoch{epo}.pth')
        torch.save(fusion_model.state_dict(), fusion_model_file)
        logger.info("Fusion Model_WO_CMFF after epoch {} Save to: {}".format(epo,fusion_model_file))
    fusion_model_file = os.path.join(model_pth, 'fusion_model_it2300.pth')
    torch.save(fusion_model.state_dict(), fusion_model_file)
    logger.info("Fusion Model_WO_CMFF Save to: {}".format(fusion_model_file))
    logger.info('\n')




def run_fusion(type='MSRS'):

    fusion_model_path = 'model/Fusion/fusion_model_WOViT_it1000.pth'
    fused_dir = os.path.join('./result', type, 'MSRS')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusion_model = eval('WO_ViT')(output=1)

    fusion_model.eval()
    n_classes = 9
    net = BiSeNet(n_classes=n_classes)
    Seg_model_path = './model/Fusion/model_final.pth'
    net.load_state_dict(torch.load(Seg_model_path))
    net.cuda(args.gpu)
    net.eval()
    print("SEGParams(M): %.3f" % (params_count(net) / (1000 ** 2)))

    if args.gpu >= 0:
        fusion_model.cuda(args.gpu)
    fusion_model.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = Fusion_dataset2(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            out, mid = net(images_vis)

            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(images_vis)
            vi_Y = vi_Y.cuda(args.gpu)
            vi_Cb = vi_Cb.cuda(args.gpu)
            vi_Cr = vi_Cr.cuda(args.gpu)
            logits = fusion_model(vi_Y, images_ir, out)

            fusion_image = YCbCr2RGB(logits, vi_Cb, vi_Cr)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)


            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    for i in range(1):
        train_fusion(i, logger, args)
        print("|{0} Train Fusion_WO_CMFF Model Successfully~!".format(i + 1))
        # run_fusion('MSRS')
        # print("|{0} Fusion Image Successfully~!".format(i + 1))
        # train_seg(i, logger, args)
        # print("|{0} Train Segmentation Model Successfully~!".format(i + 1))
    print("training Done!")