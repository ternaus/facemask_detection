from collections import namedtuple

import timm
from torch import nn
from torch.utils import model_zoo

Facemask_classifier = namedtuple("Facemask_classifier", ["url", "model"])

models = {
    "tf_efficientnet_b0_ns_2020-07-29": Facemask_classifier(
        url="https://github.com/ternaus/facemask_detection/releases/download/0.0.1/tf_efficientnet_b0_ns_2020-07-29-ffdde352.zip",  # noqa: E501
        model=timm.create_model(model_name="tf_efficientnet_b0_ns", num_classes=1),
    )
}


def get_model(model_name: str, use_sigmoid: bool = True) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")

    model.load_state_dict(state_dict)
    if use_sigmoid:
        model = nn.Sequential(model, nn.Sigmoid())
    return model
