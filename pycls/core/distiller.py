import torch
import torch.nn as nn
import torch.nn.functional as F
from pycls.core.config import cfg



class WSLDistiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(WSLDistiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

        self.T = 2
        self.alpha = 2.5

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax()

        self.hard_loss = nn.CrossEntropyLoss()
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.ratio_lower = torch.zeros(1)


    def forward(self, x, label):

        fc_t = self.t_net(x)
        fc_s = self.s_net(x)

        s_input_for_softmax = fc_s / self.T
        t_input_for_softmax = fc_t / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = -torch.sum(t_soft_label * self.logsoftmax(s_input_for_softmax), 1, keepdim=True)

        fc_s_auto = fc_s.detach()
        fc_t_auto = fc_t.detach()
        log_softmax_s = self.logsoftmax(fc_s_auto)
        log_softmax_t = self.logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
        softmax_loss_s = -torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = -torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        focal_weight = torch.max(focal_weight, self.ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss)

        hard_loss = self.hard_loss(fc_s, label)

        loss = hard_loss + self.alpha * soft_loss

        return fc_s, loss
