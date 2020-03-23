from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import resnet
from .deeplabv3_emb import DeepLabV3Emb, DeepLabHeadEmb

__all__ = ['deeplabv3_resnet50', 'deeplabv3_resnet101']


model_urls = {
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}


def _segm_resnet_emb(name, backbone_name, num_classes, aux, args, pretrained_backbone=True, embedding_model='mean', aspp_option='v3'):

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {'deeplabv3': (DeepLabHeadEmb, DeepLabV3Emb)}

    inplanes = 2048

    classifier = model_map[name][0](inplanes, num_classes, aspp_option, args)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, args, embedding_model, aspp_option, **kwargs):
    if pretrained:
        aux_loss = True
    # model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    # TODO: I changed this to see if the embedding thing works
    model = _segm_resnet_emb(arch_type, backbone, num_classes, aux_loss, args, embedding_model=embedding_model, aspp_option=aspp_option, **kwargs)

    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, embedding_model='mean', aspp_option='v3', args=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss, args, embedding_model, aspp_option, **kwargs)
