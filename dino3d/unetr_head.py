import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock


class UNETRHead(nn.Module):
    def __init__(self, feature_model, image_size, num_classes, input_channels=1, feature_size=32):
        super().__init__()
        self.feature_model = feature_model
        hidden_size = feature_model.num_features
        self.feat_size = [image_size // feature_model.patch_size] * 3
        self.hidden_size = hidden_size

        self.encoder1 = UnetrBasicBlock(3, input_channels, feature_size, kernel_size=3, stride=1,
                                        norm_name="instance", res_block=True)
        self.encoder2 = UnetrPrUpBlock(3, hidden_size, feature_size * 2, num_layer=2, kernel_size=3, stride=1,
                                       upsample_kernel_size=2, norm_name="instance", conv_block=True, res_block=True)
        self.encoder3 = UnetrPrUpBlock(3, hidden_size, feature_size * 4, num_layer=1, kernel_size=3, stride=1,
                                       upsample_kernel_size=2, norm_name="instance", conv_block=True, res_block=True)
        self.encoder4 = UnetrPrUpBlock(3, hidden_size, feature_size * 8, num_layer=0, kernel_size=3, stride=1,
                                       upsample_kernel_size=2, norm_name="instance", conv_block=True, res_block=True)
        self.decoder5 = UnetrUpBlock(3, hidden_size, feature_size * 8, kernel_size=3, upsample_kernel_size=2,
                                     norm_name="instance", res_block=True)
        self.decoder4 = UnetrUpBlock(3, feature_size * 8, feature_size * 4, kernel_size=3, upsample_kernel_size=2,
                                     norm_name="instance", res_block=True)
        self.decoder3 = UnetrUpBlock(3, feature_size * 4, feature_size * 2, kernel_size=3, upsample_kernel_size=2,
                                     norm_name="instance", res_block=True)
        self.decoder2 = UnetrUpBlock(3, feature_size * 2, feature_size, kernel_size=3, upsample_kernel_size=2,
                                     norm_name="instance", res_block=True)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=num_classes)

    def proj_feat(self, x):
        x = x.view(x.size(0), *self.feat_size, self.hidden_size)
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x_in):
        x2, x3, x4, x = self.feature_model.get_intermediate_layers(x_in, layers=[5, 11, 17, 23])
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(x2))
        enc3 = self.encoder3(self.proj_feat(x3))
        enc4 = self.encoder4(self.proj_feat(x4))
        dec3 = self.decoder5(self.proj_feat(x), enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        return self.out(self.decoder2(dec1, enc1))
