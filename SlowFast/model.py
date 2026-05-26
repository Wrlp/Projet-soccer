import torch
import torch.nn as nn
import torchvision.models.video as video_models


class SlowFastSimple(nn.Module):
    """A simplified SlowFast-like model using two 3D ResNet backbones.

    This implementation does not reproduce all lateral connections of the
    original SlowFast paper, but provides a practical two-path fusion:
    extract features from a slow backbone and a fast backbone, then concat
    and classify.
    """

    def __init__(self, num_classes, pretrained_backbones=False):
        super().__init__()
        # slow backbone
        self.slow = video_models.r3d_18(pretrained=pretrained_backbones)
        self.fast = video_models.r3d_18(pretrained=pretrained_backbones)

        # remove final fc to get feature vectors
        self.slow.fc = nn.Identity()
        self.fast.fc = nn.Identity()

        feat_dim = 512  # r3d_18 final feature dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, slow, fast):
        # slow, fast: (B, C, T, H, W)
        fs = self.slow(slow)  # (B, feat_dim)
        ff = self.fast(fast)  # (B, feat_dim)
        x = torch.cat([fs, ff], dim=1)
        logits = self.classifier(x)
        return logits


if __name__ == '__main__':
    # quick smoke
    m = SlowFastSimple(num_classes=10)
    s = torch.randn(2, 3, 8, 224, 224)
    f = torch.randn(2, 3, 64, 224, 224)
    out = m(s, f)
    print(out.shape)
