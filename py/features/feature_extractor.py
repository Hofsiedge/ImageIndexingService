import torch
from torchvision import models, transforms as T


class CNNFeatureExtractor(torch.nn.Module):
    
    basic_input_transform = T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(self, input_transform=None) -> None:
        super(CNNFeatureExtractor, self).__init__()
        mobilenet                     = models.mobilenet_v3_large(pretrained=True)
        self._features, self._avgpool = mobilenet.features, mobilenet.avgpool
        self._head                    = next(mobilenet.classifier.children())
        self.input_transform          = input_transform
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features(self.input_transform(x) if self.input_transform else x)
        out = self._avgpool(out)
        out = torch.flatten(out, 1)
        out = self._head(out)
        return out
