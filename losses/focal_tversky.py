import torch
import torch.nn as nn

class FocalTversky(nn.Module):
    def __init__(self,smooth=1):
        super(FocalTversky, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        alpha=0.7
        gamma=0.75
        smooth=1
        y_pred = torch.argmax(pred, dim=1)
        y_true_pos = torch.flatten(target)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        tv = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
        return torch.pow((1 - tv), gamma)


        # return  self.focal_tversky_loss(target,pred)

    def tversky(self,y_true, y_pred, smooth=1, alpha=0.7):
        y_pred = torch.argmax(y_pred, dim=1)
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


    def tversky_loss(self,y_true, y_pred):
        return 1 - self.tversky(y_true, y_pred)


    def focal_tversky_loss(self,y_true, y_pred, gamma=0.75):
        tv = self.tversky(y_true, y_pred)
        return torch.pow((1 - tv), gamma)