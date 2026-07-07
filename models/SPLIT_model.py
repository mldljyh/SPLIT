import torch
import torch.nn as nn
import timm

from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel
import torchvision.models as models

Transformers = [
    'CLIP-16',
    'CLIP-32',
    'XCLIP-16',
    'XCLIP-32',
    'DINO-base',
    'DINO-large',
]


class SPLIT_model(nn.Module):
    def __init__(self, encoder_type='CLIP-16', loss_type='l2'):
        super(SPLIT_model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type
        self.gamma = 8.0
        self.encoder_returns_list = False

        if encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        elif encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        elif encoder_type == 'XCLIP-16':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")

        elif encoder_type == 'XCLIP-32':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

        elif encoder_type == 'DINO-base':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")

        elif encoder_type == 'DINO-large':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")

        elif encoder_type == 'ResNet-18':
            resnet18 = models.resnet18(pretrained=True)
            modules = list(resnet18.children())[:-2]
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'VGG-16':
            vgg16 = models.vgg16(pretrained=True)
            self.encoder = vgg16.features.eval()

        elif encoder_type == 'EfficientNet-b4':
            efficientnet_b4 = models.efficientnet_b4(pretrained=True)
            self.encoder = efficientnet_b4.features.eval()

        elif encoder_type == 'MobileNet-v3':
            mobilenetv3 = timm.create_model(
                'mobilenetv3_large_100',
                pretrained=True,
                features_only=True,
                out_indices=(-1,),
            )
            self.encoder = mobilenetv3.eval()
            self.encoder_returns_list = True

    def _compute_components(self, x, return_global=True):
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)

        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
        else:
            patch_tokens = self.encoder(images)
            if self.encoder_returns_list or isinstance(patch_tokens, (list, tuple)):
                patch_tokens = patch_tokens[-1]
            if len(patch_tokens.shape) == 4:
                patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        if len(patch_tokens.shape) == 2:
            patch_tokens = patch_tokens.unsqueeze(1)

        _bt, num_patches, d = patch_tokens.shape

        patch_tokens_ttr = patch_tokens.reshape(b, t, num_patches, d).permute(0, 2, 1, 3).reshape(b * num_patches, t, d)
        ttr_val = self.ttr(patch_tokens_ttr).reshape(b, num_patches)

        grid_size = int(num_patches**0.5)
        patch_tokens_spatial = patch_tokens.reshape(b, t, grid_size, grid_size, d)

        motion = patch_tokens_spatial[:, 1:, :, :, :] - patch_tokens_spatial[:, :-1, :, :, :]
        grad_x = motion[:, :, :, :-1, :] - motion[:, :, :, 1:, :]
        grad_y = motion[:, :, :-1, :, :] - motion[:, :, 1:, :, :]

        mag_grad_x = torch.norm(grad_x, p=2, dim=-1)
        mag_grad_y = torch.norm(grad_y, p=2, dim=-1)
        mean_lsmi = (mag_grad_x.mean(dim=(1, 2, 3)) + mag_grad_y.mean(dim=(1, 2, 3))) / 2.0

        outputs_global = None
        if return_global:
            outputs_global = patch_tokens.mean(dim=(1, 2)).reshape(b, t, -1).mean(dim=1)
        return outputs_global, ttr_val, mean_lsmi

    def compute_score(self, mean_ttr, mean_lsmi):
        return (mean_ttr ** self.gamma) * mean_lsmi

    def forward(self, x):
        outputs_global, ttr_val, mean_lsmi = self._compute_components(x, return_global=True)
        mean_ttr = ttr_val.mean(dim=1)
        score = self.compute_score(mean_ttr, mean_lsmi)
        return outputs_global, score, score

    def forward_components(self, x, return_global=True):
        outputs_global, ttr_val, mean_lsmi = self._compute_components(x, return_global=return_global)
        sorted_ttr, _ = torch.sort(ttr_val, dim=1)
        return outputs_global, sorted_ttr, mean_lsmi

    def forward_score(self, x):
        _outputs_global, ttr_val, mean_lsmi = self._compute_components(x, return_global=False)
        mean_ttr = ttr_val.mean(dim=1)
        return self.compute_score(mean_ttr, mean_lsmi)

    @torch.no_grad()
    def forward_ttr_l1l2(self, x):
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)

        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
        else:
            patch_tokens = self.encoder(images)
            if self.encoder_returns_list or isinstance(patch_tokens, (list, tuple)):
                patch_tokens = patch_tokens[-1]
            if len(patch_tokens.shape) == 4:
                patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        if len(patch_tokens.shape) == 2:
            patch_tokens = patch_tokens.unsqueeze(1)

        _bt, num_patches, d = patch_tokens.shape
        patch_tokens_ttr = (
            patch_tokens.reshape(b, t, num_patches, d)
            .permute(0, 2, 1, 3)
            .reshape(b * num_patches, t, d)
        )
        ttr_val, l1_val, l2_val = self._ttr_l1_l2(patch_tokens_ttr)
        return (
            ttr_val.reshape(b, num_patches),
            l1_val.reshape(b, num_patches),
            l2_val.reshape(b, num_patches),
        )

    def ttr(self, outputs):
        ttr_val, _l1, _l2 = self._ttr_l1_l2(outputs)
        return ttr_val

    def _ttr_l1_l2(self, outputs):
        diffs = outputs[:, 1:, :] - outputs[:, :-1, :]
        l1 = torch.sum(torch.norm(diffs, p=2, dim=-1), dim=1)
        diffs_k2 = outputs[:, 2:, :] - outputs[:, :-2, :]
        l2 = torch.sum(torch.norm(diffs_k2, p=2, dim=-1), dim=1)

        t_seq = outputs.shape[1]
        L1 = l1
        L2 = (l2 / 2) * (t_seq - 1) / (t_seq - 2)

        ttr_val = (torch.log(L1 + 1e-8) - torch.log(L2 + 1e-8)) / 0.693147
        return ttr_val, L1, L2
