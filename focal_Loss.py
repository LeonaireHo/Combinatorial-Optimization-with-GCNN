import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):  
    def __init__(self, alpha=.25, gamma=2,is_cuda = True):
            super(FocalLoss, self).__init__() 
            self.alpha = torch.tensor([alpha, 1-alpha])
            if is_cuda:
                self.alpha = self.alpha.cuda()        
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            targets = targets.type(torch.long)        
            at = self.alpha.gather(0, targets.data.view(-1))        
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()


