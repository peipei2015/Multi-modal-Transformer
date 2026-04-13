import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model.HolisticAttention import HA
from .functions import BasicConv2d, Pred_decoder_bbsnet
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key1 = nn.Linear(n_embd, n_embd)
        self.query1 = nn.Linear(n_embd, n_embd)
        self.value1 = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop1 = nn.Dropout(attn_pdrop)
        self.resid_drop1 = nn.Dropout(resid_pdrop)
        # output projection
        self.proj1 = nn.Linear(n_embd, n_embd)

        # key, query, value projections for all heads
        self.key2 = nn.Linear(n_embd, n_embd)
        self.query2 = nn.Linear(n_embd, n_embd)
        self.value2 = nn.Linear(n_embd, n_embd)
        self.attn_drop2 = nn.Dropout(attn_pdrop)
        self.resid_drop2 = nn.Dropout(resid_pdrop)
        # output projection
        self.proj2 = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, d):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k1 = self.key1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q1 = self.query1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v1 = self.value1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        k2 = self.key2(d).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q2 = self.query2(d).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v2 = self.value2(d).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att1 = (q2 @ k1.transpose(-2, -1)) * (1.0 / math.sqrt(k1.size(-1)))
        att1 = F.softmax(att1, dim=-1)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att2 = (q1 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(k2.size(-1)))
        att2 = F.softmax(att2, dim=-1)

        att1 = self.attn_drop1(att1)
        y1 = att1 @ v1  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y1 = y1.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y1 = self.resid_drop1(self.proj1(y1))

        att2 = self.attn_drop2(att2)
        y2 = att2 @ v2  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y2 = y2.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y2 = self.resid_drop2(self.proj2(y2))


        return y1+y2

class SelfAttentionS(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super(SelfAttentionS, self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key1 = nn.Linear(n_embd, n_embd)
        self.query1 = nn.Linear(n_embd, n_embd)
        self.value1 = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop1 = nn.Dropout(attn_pdrop)
        self.resid_drop1 = nn.Dropout(resid_pdrop)
        # output projection
        self.proj1 = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k1 = self.key1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q1 = self.query1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v1 = self.value1(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / math.sqrt(k1.size(-1)))
        att1 = F.softmax(att1, dim=-1)

        att1 = self.attn_drop1(att1)
        y1 = att1 @ v1  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y1 = y1.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y1 = self.resid_drop1(self.proj1(y1))
        return y1

class aBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super(aBlock, self).__init__()
        self.ln0 = nn.LayerNorm(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttentionS(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn2 = SelfAttentionS(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, d):
        B, T, C = x.size()
        x1 = x+self.attn1(x)
        d1 = d+self.attn2(d)
        x = x + self.attn(self.ln1(x1), self.ln0(d1))
        x = x + self.mlp(self.ln2(x))

        return x

class TransBlock(nn.Module):
    def __init__(self, n_embd, vert_anchors=11, horz_anchors=11, n_head=8, block_exp=1, embd_pdrop=0.5, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransBlock, self).__init__()
        self.n_embd = n_embd
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        # positional embedding parameter (learnable), image + lidar
        self.x_pos_emb = nn.Parameter(
            torch.zeros(1, vert_anchors * horz_anchors, n_embd))
        self.d_pos_emb = nn.Parameter(
            torch.zeros(1, vert_anchors * horz_anchors, n_embd))

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = aBlock(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x_tensor, d_tensor):
        raw_image_tensor = x_tensor
        x_tensor = F.upsample(x_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear',
                                          align_corners=True)
        d_tensor = F.upsample(d_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear',
                                  align_corners=True)

        bz = x_tensor.shape[0]
        h, w = x_tensor.shape[2:4]

        # forward the image model for token embeddings
        x_tensor = x_tensor.view(bz, 1, -1, h, w)
        d_tensor = d_tensor.view(bz, 1, -1, h, w)

        # pad token embeddings along number of tokens dimension
        x_token_embeddings = x_tensor.permute(0, 1, 3, 4, 2).contiguous()
        x_token_embeddings = x_token_embeddings.view(bz, -1, self.n_embd)

        d_token_embeddings = d_tensor.permute(0, 1, 3, 4, 2).contiguous()
        d_token_embeddings = d_token_embeddings.view(bz, -1, self.n_embd)


        x = self.drop(self.x_pos_emb + x_token_embeddings)
        d = self.drop(self.d_pos_emb + d_token_embeddings)

        x = self.blocks(x,d)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        x = x.view(bz, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x.contiguous().view(bz, -1, h, w)

        image_tensor_out = F.upsample(image_tensor_out, size=(raw_image_tensor.shape[2], raw_image_tensor.shape[3]), mode='bilinear', align_corners=True) #raw_image_tensor +

        return image_tensor_out

class Pred_endecoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Pred_endecoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.depth_net = models.resnet18(pretrained=True)

        self.depth_layer1 = nn.Conv2d(64, 256, 1, 1, 0)
        self.depth_layer2 = nn.Conv2d(128, 512, 1, 1, 0)
        self.depth_layer3 = nn.Conv2d(256, 1024, 1, 1, 0)
        self.depth_layer4 = nn.Conv2d(512, 2048, 1, 1, 0)

        self.rgb_layer1 = nn.Conv2d(256, 128, 1, 1, 0)
        self.rgb_layer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.rgb_layer3 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.rgb_layer4 = nn.Conv2d(2048, 512, 1, 1, 0)

        block_exp = 4
        n_head = 4
        embd_pdrop = 0.1
        resid_pdrop = 0.1
        attn_pdrop = 0.1
        self.transformer0 = TransBlock(n_embd=64,
                                n_head=n_head,
                                block_exp=block_exp,
                                vert_anchors=32,
                                horz_anchors=32,
                                embd_pdrop=embd_pdrop,
                                attn_pdrop=attn_pdrop,
                                resid_pdrop=resid_pdrop)
        self.transformer1 = TransBlock(n_embd=256,
                                n_head=n_head,
                                block_exp=block_exp,
                                vert_anchors=16,
                                horz_anchors=16,
                                embd_pdrop=embd_pdrop,
                                attn_pdrop=attn_pdrop,
                                resid_pdrop=resid_pdrop)
        self.transformer2 = TransBlock(n_embd=512,
                                n_head=n_head,
                                block_exp=block_exp,
                                vert_anchors=16,
                                horz_anchors=16,
                                embd_pdrop=embd_pdrop,
                                attn_pdrop=attn_pdrop,
                                resid_pdrop=resid_pdrop)
        self.transformer3 = TransBlock(n_embd=1024,
                                n_head=n_head,
                                block_exp=block_exp,
                                vert_anchors=8,
                                horz_anchors=8,
                                embd_pdrop=embd_pdrop,
                                attn_pdrop=attn_pdrop,
                                resid_pdrop=resid_pdrop)
        self.transformer4 = TransBlock(n_embd=2048,
                                n_head=n_head,
                                block_exp=block_exp,
                                vert_anchors=8,
                                horz_anchors=8,
                                embd_pdrop=embd_pdrop,
                                attn_pdrop=attn_pdrop,
                                resid_pdrop=resid_pdrop)


        self.sod_dec = Pred_decoder_bbsnet(channel)
        self.sod_dec2 = Pred_decoder_bbsnet(channel)

        self.upsample05 = nn.Upsample(scale_factor=1.0/2, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=1.0/4, mode='bilinear', align_corners=True)
        self.upsample0125 = nn.Upsample(scale_factor=1.0/8, mode='bilinear', align_corners=True)
        self.upsample00625 = nn.Upsample(scale_factor=1.0/16, mode='bilinear', align_corners=True)
        self.upsample00312 = nn.Upsample(scale_factor=1.0/32, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, depth_x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) #x torch.Size([10, 64, 88, 88])
        d = self.depth_net.conv1(torch.cat((depth_x, depth_x, depth_x), dim=1))
        d = self.depth_net.bn1(d)
        d = self.depth_net.relu(d)
        d = self.depth_net.maxpool(d)
        x0_1 = self.transformer0(x, d)+x

        x1 = self.resnet.layer1(x0_1)
        d1 = self.depth_net.layer1(d)
        x1_1 = self.transformer1(x1, self.depth_layer1(d1))+x1

        x2 = self.resnet.layer2(x1_1)
        d2 = self.depth_net.layer2(d1)
        x2_1 = self.transformer2(x2, self.depth_layer2(d2))+x2

        x3 = self.resnet.layer3(x2_1)
        d3 = self.depth_net.layer3(d2)
        x3_1 = self.transformer3(x3, self.depth_layer3(d3))+x3

        x4 = self.resnet.layer4(x3_1)
        d4 = self.depth_net.layer4(d3)
        x4_1 = self.transformer4(x4, self.depth_layer4(d4))+x4


        x1_1 = self.rgb_layer1(x1_1) #x1 torch.Size([10, 128, 88, 88])
        x2_1 = self.rgb_layer2(x2_1) #x2 torch.Size([10, 256, 44, 44])
        x3_1 = self.rgb_layer3(x3_1) #x3 torch.Size([10, 512, 22, 22])
        x4_1 = self.rgb_layer4(x4_1) #x4 torch.Size([10, 512, 11, 11])

        att, sal = self.sod_dec(x0_1, x1_1, x2_1, x3_1, x4_1)
        return self.upsample8(att), sal


    def initialize_weights(self):
        print('initialize with params of pretrained model')
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_3' in k:
                name = k.split('_3')[0] + k.split('_3')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
