from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


# segmentation model with embedding 
class _SimpleSegmentationModelEmb(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None, inference_backbone=False):
        super(_SimpleSegmentationModelEmb, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.inference_backbone = inference_backbone

    def forward(self, x, emb):

        if self.inference_backbone:

            with torch.no_grad():
                input_shape = x.shape[-2:]
                # contract: features is a dict of tensors
                features = self.backbone(x)

                result = OrderedDict()
                x = features["out"]

        else:

            input_shape = x.shape[-2:]
            # contract: features is a dict of tensors
            features = self.backbone(x)
            result = OrderedDict()
            x = features["out"]

        x, v, l = self.classifier(x, emb)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        # return result
        return result, v, l