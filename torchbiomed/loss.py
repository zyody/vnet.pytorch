import pdb

import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

class DiceLoss(Function):
    '''
    Compute energy based on dice coefficient.
    Aims to maximize dice coefficient.
    '''
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.00001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
#       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        dice = 2*intersect / (union + eps)
        print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} pred_sum: {:.0f} dice_coefficient: {:.7f}'.format(
            union, intersect, target_sum, result_sum, dice))
        out = torch.FloatTensor(1).fill_(dice)
        if input.is_cuda:
            out = out.cuda() # added by Chao.
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        part1 = torch.div(target, union)
        part2_2 = intersect/(union*union)
        part2 = torch.mul(input[:, 1], part2_2)
        dDice = torch.add(torch.mul(part1, 2), torch.mul(part2, -4))
        # print("grad_output:{}".format(grad_output))
        # why fix grad_output:tensor([1.], device='cuda:3')??? By Chao.
        grad_input = torch.cat((torch.mul(dDice, grad_output[0]).view(-1,1), torch.mul(dDice, -grad_output[0]).view(-1,1)), 1)
        return grad_input, None

def dice_loss(input, target):
    return DiceLoss()(input, target)

def dice_error(input, target):
    eps = 0.00001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    dice = 2*intersect / (union + eps)
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return dice
