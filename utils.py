import torch
import numpy as np
from PIL import Image

import re


def RGB2YCrCb(rgb_image):

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):

    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out


def tensor2img(img, is_norm=True):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  if is_norm:
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)


def save_img_single(img, name, is_norm=True):
  img = tensor2img(img, is_norm=True)
  img = Image.fromarray(img)
  img.save(name)




def natural_sort_key(s):

    parts = re.split(r'(\d+)', s)
    sorted_key = []
    for part in parts:
        if part.isdigit():
            sorted_key.append(int(part))
        else:
            sorted_key.append(part)

    return sorted_key


def natsorted(iterable):

    return sorted(iterable, key=natural_sort_key)

if __name__ == '__main__':
    files = ["file2.txt", "file10.txt", "file1.txt", "file23.txt", "file2a.txt"]
    sorted_files = natsorted(files)
    print(sorted_files)













class SegmentationMetric(object):
    def __init__(self, numClass, device):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2).to(device)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):

        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean()
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = torch.diag(self.confusionMatrix)
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU < float('inf')].mean()
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):

        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):

        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        with torch.no_grad():
            self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))


def compute_results(conf_total):
    n_class = conf_total.shape[0]
    consider_unlabeled = True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class):
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] = np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(
                conf_total[start_index:, cid].sum())  # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(
                conf_total[cid, start_index:].sum())  # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[
                                                                                                            start_index:,
                                                                                                            cid].sum() -
                                                                      conf_total[cid, cid]))  # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class