import torch
from torch import nn
from torch.nn import functional as F


from ._utils import _SimpleSegmentationModelEmb


__all__ = ["DeepLabV3Emb"]


class DeepLabV3Emb(_SimpleSegmentationModelEmb):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

# new class to consider the embedding
class DeepLabHeadEmb(nn.Sequential):
    def __init__(self, in_channels, num_classes, aspp_option, args):
        super(DeepLabHeadEmb, self).__init__()

        self.aspp = ASPP_v4(in_channels, [12, 24, 36], args)
        self.conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, emb):

        x, v, l = self.aspp(x, emb)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.conv_2(x), v, l

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP_v4(nn.Module):
    def __init__(self, in_channels, atrous_rates, args):
        super(ASPP_v4, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.embedding_project = nn.Sequential(nn.Linear(768, out_channels), nn.BatchNorm1d(out_channels))

        self.project = nn.Sequential(
            # we add one out_channels for the embedding
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.multiply_feats = args.multiply_feats
        self.addition = args.addition

        if self.multiply_feats or self.addition:
            out_channels_multimodal = out_channels
        else:
            out_channels_multimodal = 2*out_channels

        self.project_final = nn.Sequential(nn.Conv2d(out_channels_multimodal, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))


    # main difference is that now it accepts one input more when forwarding
    def forward(self, x, embedding):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res = torch.cat(res, dim=1)
        visual_feats = self.project(res)

        p_emb = self.embedding_project(embedding)
        p_emb_s = p_emb
        # repeat i every spatial position
        p_emb = p_emb.unsqueeze(-1)
        p_emb = p_emb.repeat(1, 1, res[0].size(-2)*res[0].size(-1))
        p_emb = p_emb.resize(p_emb.size(0), p_emb.size(1), visual_feats[0].size(-2), visual_feats[0].size(-1))

        # multiply feats
        if self.multiply_feats:
            res = torch.mul(visual_feats, p_emb)
        # addition of feats
        elif self.addition:       
            res = torch.add(visual_feats, p_emb)/2
        # concatanation of feats
        else:
            res = torch.cat([visual_feats, p_emb], dim=1)

        res = self.project_final(res)

        return res, visual_feats, p_emb

