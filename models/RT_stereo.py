from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np

norm_layer2d = nn.BatchNorm2d 
norm_layer3d = nn.BatchNorm3d 


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, groups = 1):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, groups = groups, bias=False),
                         norm_layer2d(out_planes))

class featexchange(nn.Module):
    def __init__(self):
        super(featexchange, self).__init__()

        self.x2_fusion = nn.Sequential(nn.ReLU(),convbn(4, 4, 3, 1, 1, 1, 1),
                                       nn.ReLU(),nn.Conv2d(4, 4, 3, 1, 1, bias=False))
        self.upconv4 = nn.Sequential(nn.Conv2d(8, 4, 1, 1, 0, bias=False),
                                     norm_layer2d(4)) 
        self.upconv8 = nn.Sequential(nn.Conv2d(20, 4, 1, 1, 0, bias=False),
                                     norm_layer2d(4)) 

        self.x4_fusion = nn.Sequential(nn.ReLU(),convbn(8, 8, 3, 1, 1, 1, 1),
                                       nn.ReLU(),nn.Conv2d(8, 8, 3, 1, 1, bias=False))
        self.downconv4 = nn.Sequential(nn.Conv2d(4, 8, 3, 2, 1, bias=False),
                                       norm_layer2d(8))
        self.upconv8_2 = nn.Sequential(nn.Conv2d(20, 8, 1, 1, 0, bias=False),
                                     norm_layer2d(8))

        self.x8_fusion = nn.Sequential(nn.ReLU(),convbn(20, 20, 3, 1, 1, 1, 1),
                                       nn.ReLU(),nn.Conv2d(20, 20, 3, 1, 1, bias=False))
        self.downconv81 = nn.Sequential(nn.Conv2d(8, 20, 3, 2, 1, bias=False),
                                       norm_layer2d(20))
        self.downconv82 = nn.Sequential(nn.Conv2d(8, 20, 3, 2, 1, bias=False),
                                       norm_layer2d(20))

    def forward(self, x2, x4, x8, attention):

        A = torch.split(attention,[4,8,20],dim=1)

        x4tox2 = self.upconv4(F.upsample(x4, (x2.size()[2],x2.size()[3])))
        x8tox2 = self.upconv8(F.upsample(x8, (x2.size()[2],x2.size()[3])))
        fusx2  = x2 + x4tox2 + x8tox2
        fusx2  = self.x2_fusion(fusx2)*A[0].contiguous()+fusx2

        x2tox4 = self.downconv4(x2)
        x8tox4 = self.upconv8_2(F.upsample(x8, (x4.size()[2],x4.size()[3])))
        fusx4  = x4 + x2tox4 + x8tox4 
        fusx4  = self.x4_fusion(fusx4)*A[1].contiguous()+fusx4

        x2tox8 = self.downconv81(x2tox4)
        x4tox8 = self.downconv82(x4)
        fusx8  = x8 + x2tox8 + x4tox8
        fusx8  = self.x8_fusion(fusx8)*A[2].contiguous()+fusx8

        return fusx2, fusx4, fusx8

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 1
        self.firstconv = nn.Sequential(nn.Conv2d(3, 3, 3, 2, 1, bias=False),
                                       nn.Conv2d(3, 3, 3, 2, 1, bias=False),
		                               nn.BatchNorm2d(3),                                                                              
                                       nn.ReLU(),
				                       nn.Conv2d(3, 4, 1, 1, 0, bias=False),
                                       convbn(4, 4, 3, 1, 1, 1, 4),
                                       nn.ReLU(),
				                       nn.Conv2d(4, 4, 1, 1, 0, bias=False),                                     
                                       convbn(4, 4, 3, 1, 1, 1, 4)) # 1/4

        self.stage2 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(4, 8, 1, 1, 0, bias=False),
                                    convbn(8, 8, 3, 2, 1, 1, 8),
                                    nn.ReLU(),
				                    nn.Conv2d(8, 8, 1, 1, 0, bias=False),
                                    convbn(8, 8, 3, 1, 1, 1, 8)) # 1/8

        self.stage3 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(8, 20, 1, 1, 0, bias=False),
                                    convbn(20, 20, 3, 2, 1, 1, 20),
                                    nn.ReLU(),
				                    nn.Conv2d(20, 20, 1, 1, 0, bias=False),
                                    convbn(20, 20, 3, 1, 1, 1,20)) #1/16
                
        self.stage4 = nn.Sequential(nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(20, 10, 1, 1, 0, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(10, 32, 1, 1, 0, bias=True),
                                    nn.Sigmoid(),
                                    ) 
        
        self.fusion = featexchange()
        

    def forward(self, x):
        #stage 1# 1x
        out_s1 = self.firstconv(x)
        out_s2 = self.stage2(out_s1) 
        out_s3 = self.stage3(out_s2)        
        attention = self.stage4(out_s3)        
        out_s1, out_s2, out_s3 = self.fusion(out_s1, out_s2, out_s3, attention)          
        return [out_s3, out_s2, out_s1]

def batch_relu_conv3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1, bn3d=True):
    if bn3d:
        return nn.Sequential(
            norm_layer3d(in_planes),
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))

def post_3dconvs(layers, channels):
    net  = [nn.Conv3d(1, channels, kernel_size=3, padding=1, stride=1, bias=False)]
    net += [batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net += [batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)

class RTStereoNet(nn.Module):
    def __init__(self, maxdisp):
        super(RTStereoNet, self).__init__()

        self.feature_extraction = feature_extraction()
        self.maxdisp = maxdisp
        self.volume_postprocess = []
        
        layer_setting = [8,4,4]
        for i in range(3):
            net3d = post_3dconvs(3, layer_setting[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.ModuleList(self.volume_postprocess)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            vgrid = grid.cuda()
        #vgrid = grid
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
        b,c,h,w = feat_l.size()
        cost = torch.zeros(b, 1, maxdisp//stride, h, w).cuda().requires_grad_(False)
        for i in range(0, maxdisp, stride):
            if i > 0:
                cost[:, :, i//stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], p=1, dim = 1,keepdim=True)
            else:
                cost[:, :, i//stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], p=1,  dim =1,keepdim=True)
        return cost.contiguous()

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        b,c,h,w = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,h,w)
        temp_array = np.tile(np.array(range(-maxdisp + 1, maxdisp)), b) * stride
        batch_shift = torch.Tensor(np.reshape(temp_array, [len(temp_array), 1, 1, 1])).cuda().requires_grad_(False)
        batch_disp = batch_disp - batch_shift
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,c,h,w)
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,c,h,w)
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1,keepdim=True)
        cost = cost.view(b,1 ,-1, h, w).contiguous()
        return cost

    def forward(self, left, right):

        img_size = left.size()

        feats_l = self.feature_extraction(left)
        feats_r = self.feature_extraction(right)
        pred = []
        for scale in range(len(feats_l)):
            if scale > 0:
                wflow = F.upsample(pred[scale-1], (feats_l[scale].size(2), feats_l[scale].size(3)),
                                   mode='bilinear') * feats_l[scale].size(2) / img_size[2]
                cost = self._build_volume_2d3(feats_l[scale], feats_r[scale], 3, wflow, stride=1)
            else:
                cost = self._build_volume_2d(feats_l[scale], feats_r[scale], 12, stride=1)

            #cost = torch.unsqueeze(cost, 1)
            cost = self.volume_postprocess[scale](cost)
            cost = cost.squeeze(1)
            if scale == 0:
                pred_low_res = disparityregression2(0, 12)(F.softmax(cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up)
            else:
                pred_low_res = disparityregression2(-2, 3, stride=1)(F.softmax(cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2) 
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up+pred[scale-1]) #
        if self.training:
            return pred[0],pred[1],pred[2]
        else:
            return pred[-1]

class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride).view(1, -1, 1, 1).type(torch.FloatTensor).cuda().requires_grad_(False)
    def forward(self, x):
        out = torch.sum(x * self.disp, 1, keepdim=True)
        return out
