import torch.nn as nn
import torch
import torch.nn.functional as F
# from .DeepLabV3 import DeepLabV3
try:
    from .DeepLabV3 import DeepLabV3
except ImportError:
    from DeepLabV3 import DeepLabV3
import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from einops import rearrange


class VGD_Network(nn.Module):
    def __init__(self, pretrained_path=None, num_classes=1): 
        super(VGD_Network, self).__init__()
        self.encoder = DeepLabV3()
        
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            print(f"Load checkpoint:{pretrained_path}")
            self.encoder.load_state_dict(checkpoint['model'])

        self.ra_attention_low = Relation_Attention(in_channels=256, out_channels=256)
        self.ra_attention_cross = Relation_Attention(in_channels=256, out_channels=256)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.final_pre = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1) 
        )

        self.refine_encoder1 = RefNet_encoder(323, 128)
        self.refine_decoder1 = RefNet_decoder()

        self.attn1_new = frame_attention(dim=96)
        self.contrast1_new = ContrastModule(planes=128) 

        initialize_weights(self.ra_attention_low, self.ra_attention_cross, self.project, 
                           self.final_pre, 
                           self.refine_encoder1, self.refine_decoder1, self.attn1_new,
                           self.contrast1_new
                           )

    def forward(self, input1, input2, input3):
        input_size = input1.size()[2:]
        exemplar0, low_exemplar1, exemplar2, exemplar3, exemplar4, exemplar = self.encoder(input1)
        query0, low_query1, query2, query3, query4, query = self.encoder(input2)
        other0, low_other1, other2, other3, other4, other = self.encoder(input3)

        # low_exemplar, exemplar = self.encoder(input1)
        # low_query, query = self.encoder(input2)
        # low_other, other = self.encoder(input3)

        # print("low_exemplar.shape: ", low_exemplar.shape)
        # print("low_query.shape: ", low_query.shape)
        # low_exemplar.shape:  torch.Size([2, 256, 104, 104])
        # low_query.shape:  torch.Size([2, 256, 104, 104])
        # exemplar.shape:  torch.Size([2, 256, 26, 26])
        # query.shape:  torch.Size([2, 256, 26, 26])

        #ehnance low level feature
        low_exemplar, low_query = self.ra_attention_low(low_exemplar1, low_query1)

        x1, x2 = self.ra_attention_cross(exemplar, query)
        
        x1 = F.interpolate(x1, size=low_exemplar.shape[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=low_query.shape[2:], mode='bilinear', align_corners=False)
        x3 = F.interpolate(other, size=low_other1.shape[2:], mode='bilinear', align_corners=False)
        fuse_exemplar = torch.cat([x1, self.project(low_exemplar)], dim=1)
        fuse_query = torch.cat([x2, self.project(low_query)], dim=1)
        fuse_other = torch.cat([x3, self.project(low_other1)], dim=1)

        exemplar_pre = self.final_pre(fuse_exemplar)
        query_pre = self.final_pre(fuse_query)
        other_pre = self.final_pre(fuse_other)  # 304 -> 256 
        exemplar_pre = F.upsample(exemplar_pre, input_size, mode='bilinear', align_corners=False) 
        query_pre = F.upsample(query_pre, input_size, mode='bilinear', align_corners=False) 
        other_pre = F.upsample(other_pre, input_size, mode='bilinear', align_corners=False) 
        
        # # ###### generate reflection
        exp_hx5 = self.refine_encoder1(exemplar0, exemplar, input1, exemplar_pre)
        query_hx5 = self.refine_encoder1(query0, query, input2, query_pre)
        other_hx5 = self.refine_encoder1(other0, other, input3, other_pre)
        ## for deformable attention
        exp_hx5 = self.contrast1_new(exp_hx5)
        query_hx5 = self.contrast1_new(query_hx5)
        other_hx5 = self.contrast1_new(other_hx5)
        exp_hx5, query_hx5, other_hx5 = self.attn1_new(exp_hx5, query_hx5, other_hx5) 
        exemplar_final, exemplar_ref = self.refine_decoder1(exemplar0, low_exemplar1, exemplar2, exemplar3, exemplar4, exp_hx5)
        query_final, query_ref = self.refine_decoder1(query0, low_query1, query2, query3, query4, query_hx5)
        other_final, other_ref = self.refine_decoder1(other0, low_other1, other2, other3, other4, other_hx5) 

        # print(exemplar_final.shape, exemplar_ref.shape)
        # print(query_final.shape, query_ref.shape)
        # print(other_final.shape, other_ref.shape)

        if self.training:
            return exemplar_pre, query_pre, other_pre, \
                exemplar_final, query_final, other_final, \
                    exemplar_ref, query_ref, other_ref
        else:
            # return exemplar_final, query_final, other_final
            return exemplar_final, exemplar_ref, exemplar_pre 
        
        
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC

        return x

class RefNet_encoder(nn.Module):
    def __init__(self,in_ch, inc_ch):
        super(RefNet_encoder, self).__init__()

        self.conv0_1 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        from networks.bra_nchw import deform_Block 
        self.deform1 = deform_Block(dim=inc_ch, num_heads=1)
        self.deform2 = deform_Block(dim=inc_ch, num_heads=1)
        self.deform3 = deform_Block(dim=inc_ch, num_heads=1)
        self.deform4 = deform_Block(dim=inc_ch, num_heads=1)


    def forward(self, exemplar0, exemplar, input1, exemplar_pre):
        exemplar_pre_small = F.interpolate(exemplar_pre, size=exemplar.shape[2:], mode='bilinear', align_corners=False)
        exemplar_feats_small = F.interpolate(exemplar0, size=exemplar.shape[2:], mode='bilinear', align_corners=False)
        image_small = F.interpolate(input1, size=exemplar.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([exemplar_feats_small, exemplar, image_small], dim=1)  # 325 channel

        hx = self.conv0_1(x)

        exemplar_pre_small = (exemplar_pre_small.sigmoid() > 0.5) * 1.0
        attn_mask = exemplar_pre_small.masked_fill(exemplar_pre_small == 0, float(-100.0)).masked_fill(exemplar_pre_small > 0, float(0.0))

        hx = self.deform1(hx, glass_mask=attn_mask)
        hx = self.deform2(hx, glass_mask=attn_mask)
        hx = self.deform3(hx, glass_mask=attn_mask)
        hx = self.deform4(hx, glass_mask=attn_mask)

        return hx


class RefNet_decoder(nn.Module):
    def __init__(self):
        super(RefNet_decoder, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(128+64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(128+96, 128, 3, padding=1),  # 64+48
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.deconv0 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64+48
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 3, padding=1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, hx0, hx1, hx2, hx3, hx4, hx5):
        hx = hx5 

        d4 = self.deconv4(torch.cat((hx, self.conv4(hx3)), 1))
        hx = self.upsample(d4)
        # hx = d4 

        d3 = self.deconv3(torch.cat((hx, self.conv3(hx2)), 1))
        hx = self.upsample(d3)

        d2 = self.deconv2(torch.cat((hx, self.conv2(hx1)), 1))
        hx = self.upsample(d2)

        d1 = self.deconv1(torch.cat((hx, hx0), 1))
        d1 = self.upsample(d1)
        output = self.deconv0(d1)

        x0, ref = torch.split(output, [1, 3], 1)
        
        return x0, ref 

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class frame_attention(nn.Module):
    def __init__(self, dim=64, num_heads=4, mlp_ratio=4):
        super(frame_attention, self).__init__()
        self.dim = dim

        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.ln_1 = LayerNorm(dim)

        self.mlp1 = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim))
        self.mlp2 = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim))

    def self_attn(self, x):
        return self.attn(x, x, x)[0]
    
    def forward(self, exemplar, query, other):
        x = torch.cat([exemplar.unsqueeze(2), query.unsqueeze(2), other.unsqueeze(2)], dim=2)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b d t h w -> (b t) d h w')
        x = x.reshape(x.shape[0], x.shape[1], -1) # (B*T, C, H*W)
        x = x.permute(2, 0, 1) # (H*W, B*T, C)
        n, bt, d = x.shape

        x = rearrange(x, 'n (b t) d -> t (b n) d', t=T)

        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp1(x)
        x = rearrange(x, 't (b n) d -> n (b t) d', n=n)

        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp2(x) 

        x = x.permute(1, 0, 2)
        x = rearrange(x, '(b t) (h w) d ->b d t h w',b=B,t=T, h=H)

        out1, out2, out3 = torch.split(x, [1, 1, 1], dim=2)
        
        return out1.squeeze(2), out2.squeeze(2), out3.squeeze(2)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, num_context=6):
        super(SELayer, self).__init__()
        self.channel = channel
        self.num_context = num_context
        self.context_channel = int(channel / num_context)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.context_attention = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, num_context, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, groups=num_context, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, groups=num_context, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        context_attention = self.context_attention(y)
        channel_attention = self.channel_attention(y)
        context_attention = context_attention.repeat(1, 1, self.context_channel, 1)
        context_attention = context_attention.view(-1, self.channel, 1, 1)
        attention = context_attention * channel_attention
        return x * attention.expand_as(x)

class ContrastModule(nn.Module):
    def __init__(self, planes):
        super(ContrastModule, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 4)

        self.local_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())
        self.context_3 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU())

        self.SELayer = SELayer(int(self.inplanes / 4 * 3))

    def forward(self, x):
        local_1 = self.local_1(x)
        
        context_1 = self.context_1(x)
        ccl_01 = local_1 - context_1

        context_2 = self.context_2(x)
        ccl_02 = local_1 - context_2

        context_3 = self.context_3(x)
        ccl_03 = local_1 - context_3

        output = torch.cat((ccl_01, ccl_02, ccl_03), 1)

        output = self.SELayer(output)
        return output


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class RAttention(nn.Module):
    '''This part of code is refactored based on https://github.com/Serge-weihao/CCNet-Pure-Pytorch. 
       We would like to thank Serge-weihao and the authors of CCNet for their clear implementation.'''
    def __init__(self,in_dim):
        super(RAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))


    def forward(self, x_exmplar, x_query):
        m_batchsize, _, height, width = x_query.size()
        proj_query = self.query_conv(x_query)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_query_LR = torch.diagonal(proj_query, 0, 2, 3)
        proj_query_RL = torch.diagonal(torch.transpose(proj_query, 2, 3), 0, 2, 3)
        # .contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_key = self.key_conv(x_exmplar)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_key_LR = torch.diagonal(proj_key, 0, 2, 3).permute(0,2,1).contiguous()
        proj_key_RL = torch.diagonal(torch.transpose(proj_key, 2, 3), 0, 2, 3).permute(0,2,1).contiguous()

        proj_value = self.value_conv(x_exmplar)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_value_LR = torch.diagonal(proj_value, 0, 2, 3)
        proj_value_RL = torch.diagonal(torch.transpose(proj_value, 2, 3), 0, 2, 3)

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)

        # energy_LR = torch.bmm(proj_query_LR, proj_key_LR)
        # energy_RL = torch.bmm(proj_query_RL, proj_key_RL)
        energy_LR = torch.bmm(proj_key_LR, proj_query_LR)
        energy_RL = torch.bmm(proj_key_RL, proj_query_RL)


        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        out_LR = self.softmax(torch.bmm(proj_value_LR, energy_LR).unsqueeze(-1))
        out_RL = self.softmax(torch.bmm(proj_value_RL, energy_RL).unsqueeze(-1))

        # print(out_H.size())
        # print(out_LR.size())
        # print(out_RL.size())


        return self.gamma_1*(out_H + out_W + out_LR + out_RL) + x_exmplar, self.gamma_2*(out_H + out_W + out_LR + out_RL) + x_query, 

class Relation_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Relation_Attention, self).__init__()
        inter_channels = in_channels // 4
        self.conv_examplar = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.conv_query = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.ra = RAttention(inter_channels)
        self.conv_examplar_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),nn.ReLU(inplace=False))
        self.conv_query_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),nn.ReLU(inplace=False))

            
    def forward(self, x_exmplar, x_query, recurrence=2):
        # print(x_exmplar.size())
        # print(x_query.size())

        x_exmplar = self.conv_examplar(x_exmplar)
        x_query = self.conv_query(x_query)
        for i in range(recurrence):
            x_exmplar, x_query = self.ra(x_exmplar, x_query)
        x_exmplar = self.conv_examplar_tail(x_exmplar)
        x_query = self.conv_query_tail(x_query)
        return x_exmplar, x_query

        # output = self.conva(x)
        # for i in range(recurrence):
        #     output = self.ra(output)
        # output = self.convb(output)
        
        # return output
        

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == '__main__':
    model = VGD_Network().cuda()
    # initialize_weights(model)
    exemplar = torch.rand(2, 3, 416, 416).cuda()
    query = torch.rand(2, 3, 416, 416).cuda()
    other = torch.rand(2, 3, 416, 416).cuda() 
    # , examplar_final, query_final, other_final
    exemplar_pre, query_pre, other_pre = model(exemplar, query, other)
    print(exemplar_pre.shape)
    print(query_pre.shape)

