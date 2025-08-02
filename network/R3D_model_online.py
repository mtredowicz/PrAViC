import torch
import torchvision
from torch import nn
from torchvision.models.video import VideoResNet, R3D_18_Weights, R2Plus1D_18_Weights, MC3_18_Weights, S3D_Weights
from torchvision.models.video import r3d_18, r2plus1d_18, mc3_18, s3d

from torch.nn import functional as F
from torch import Tensor
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        if "padding" in kwargs and kwargs.get("padding")[0] > 0:
            padding = kwargs.get("padding")
            self.padding_depth = padding[0]
            kwargs["padding"] = (0, padding[1], padding[2])
        else:
            self.padding_depth = 0
        super(CustomConv3d, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs
        )

    def forward(self, x):
        x = self.modify_input(x)
        x = super(CustomConv3d, self).forward(x)
        return x

    def modify_input(self, x):
        if self.padding_depth > 0:
            k = 2 * self.padding_depth
            first_element = x[..., 0:1, :, :].tile(1, 1, k, 1, 1)
            x = torch.cat((first_element, x), dim=-3)
        return x

class CustomConv3dv2(nn.Conv3d):
    _instance_count = 0

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(CustomConv3dv2, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs
        )
        self.id = CustomConv3dv2._instance_count
        CustomConv3dv2._instance_count += 1

    def forward(self, x):
        x = self.modify_input(x)
        x = F.conv3d(
            x, self.weight, self.bias, self.stride, (0,) * 3, self.dilation, self.groups
        )
        if self.id == 0:
            x = self.modify_output(x)
        return x

    def modify_input(self, x):
        pad = list(self._reversed_padding_repeated_twice)
        if self.padding[0] > 0:
            pad[-2] = sum(pad[-2:])
            pad[-1] = 0
        if self.padding_mode == "zeros":
            x = F.pad(x, pad, mode="constant", value=0)
        else:
            x = F.pad(x, pad, mode=self.padding_mode)
        return x

    def modify_output(self, x):
        num_remove = max(1, 1 + self.padding[0] - (self.stride[0] - 1))
        x = x[:, :, :-num_remove]

        pad = [0] * 6
        pad[-2] = num_remove
        x = F.pad(x, pad, mode="constant", value=0)
        return x


class CustomBatchNorm3d(nn.BatchNorm3d):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        depth_channel_stats=None,
    ) -> None:
        """
        Customized Batch Normalization layer for 3D input tensors.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): Small value to prevent division by zero in the normalization. Defaults to 1e-5.
            momentum (float, optional): Momentum factor for computing the running statistics. Defaults to 0.1.
            affine (bool, optional): If True, learnable affine parameters are enabled. Defaults to True.
            track_running_stats (bool, optional): If True, keep track of running statistics during training. Defaults to True.
            device (str, optional): Device on which the computations should be performed. Defaults to None.
            dtype (torch.dtype, optional): Data type of the input. Defaults to None.
            depth_channel_stats (int, optional): Depth of channels to apply statistics. Defaults to None.
        """
        super(CustomBatchNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.depth_channel_stats = depth_channel_stats

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the custom batch normalization layer.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Normalized output tensor.
        """
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        x = self.modify_input(input)
        x = F.batch_norm(
            x,
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return self.modify_output(input, x)

    def modify_input(self, x: Tensor) -> Tensor:
        """
        Modify the input tensor based on the depth_channel_stats.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Modified input tensor.
        """
        return x[:, :, : self.depth_channel_stats]

    def modify_output(self, input: Tensor, v: Tensor) -> Tensor:
        """
        Modify the output tensor based on the depth_channel_stats.

        Args:
            input (Tensor): Input tensor.
            v (Tensor): Output tensor which was calculated statistics.

        Returns:
            Tensor: Modified output tensor.
        """
        temp = input[:, :, self.depth_channel_stats :]
        if self.depth_channel_stats is not None and temp.numel():
            if self.training:
                tmp = input[:, :, : self.depth_channel_stats]
                mean = torch.mean(tmp, dim=[0, 2, 3, 4])
                var = torch.var(tmp, dim=[0, 2, 3, 4], unbiased=False)
            else:
                mean = self.running_mean
                var = self.running_var

            mean = mean.view([1, self.num_features, 1, 1, 1])
            var = var.view([1, self.num_features, 1, 1, 1])
            gamma = self.weight.view([1, self.num_features, 1, 1, 1])
            beta = self.bias.view([1, self.num_features, 1, 1, 1])

            temp = (
                torch.mul(gamma, (temp - mean))
                .div(torch.sqrt(var + self.eps))
                .add(beta)
            )
            v = torch.cat([v, temp], dim=-3)
        return v

    def extra_repr(self):
        """
        Extra representation of the custom batch normalization layer.

        Returns:
            str: Extra representation string.
        """
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}, depth_channel_stats={depth_channel_stats}".format(
                **self.__dict__
            )
        )
class CustomMaxPool3d(nn.MaxPool3d):

    def __init__(self, kernel_size, *args, **kwargs):
        super(CustomMaxPool3d, self).__init__(
             kernel_size, *args, **kwargs
        )
        if isinstance(self.padding, int):
            self.padding = (self.padding,) * 3

    def forward(self, x):
        x = self.modify_input(x)

        x = F.max_pool3d(x, self.kernel_size, self.stride,
                            (0,*self.padding[1:]), self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)
        return x

    def modify_input(self, x):
        if self.padding[0] > 0:
            pad = [0]*4 + [self.padding[0]*2, 0]
            x = F.pad(x, pad, mode="replicate")
        return x
    
dyn_depth_channel_stats = 16
def replace_layers_arch_v1(model, depth_channel_stats=None):
    global dyn_depth_channel_stats
    
    for name, module in model.named_children():
        if isinstance(module, nn.Conv3d):
            custom_conv = CustomConv3d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            if isinstance(module.stride, tuple) and module.stride[0] != 1:
                dyn_depth_channel_stats //= module.stride[0]
            elif not isinstance(module.stride, tuple) and module.stride != 1:
                dyn_depth_channel_stats //= module.stride
           
            custom_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                custom_conv.bias.data = module.bias.data.clone()
            setattr(model, name, custom_conv)
            
        elif isinstance(module, nn.BatchNorm3d):
            custom_bn = CustomBatchNorm3d(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                dyn_depth_channel_stats
            )
            if module.track_running_stats:
                custom_bn.running_mean = module.running_mean.data.clone()
                custom_bn.running_var = module.running_var.data.clone()
                custom_bn.num_batches_tracked = module.num_batches_tracked.data.clone()
            setattr(model, name, custom_bn)
        elif isinstance(module, nn.MaxPool3d):
            custom_mp = CustomMaxPool3d(
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.return_indices,
                module.ceil_mode
            )
            setattr(model, name, custom_mp)
            if isinstance(module.stride, tuple) and module.stride[0] != 1:
                dyn_depth_channel_stats //= module.stride[0]
            elif not isinstance(module.stride, tuple) and module.stride != 1:
                dyn_depth_channel_stats //= module.stride
        elif isinstance(module, nn.Module):
            replace_layers_arch_v1(module, dyn_depth_channel_stats)

def replace_layers_arch_v2(model, depth_channel_stats=None):
    global dyn_depth_channel_stats
    
    for name, module in model.named_children():
        if isinstance(module, nn.Conv3d):
            custom_conv = CustomConv3dv2(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            if isinstance(module.stride, tuple) and module.stride[0] != 1:
                dyn_depth_channel_stats //= module.stride[0]
            elif not isinstance(module.stride, tuple) and module.stride != 1:
                dyn_depth_channel_stats //= module.stride
           
            custom_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                custom_conv.bias.data = module.bias.data.clone()
            setattr(model, name, custom_conv)
            
        elif isinstance(module, nn.BatchNorm3d):
            custom_bn = CustomBatchNorm3d(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                dyn_depth_channel_stats
            )
            if module.track_running_stats:
                custom_bn.running_mean = module.running_mean.data.clone()
                custom_bn.running_var = module.running_var.data.clone()
                custom_bn.num_batches_tracked = module.num_batches_tracked.data.clone()
            setattr(model, name, custom_bn)
        elif isinstance(module, nn.MaxPool3d):
            custom_mp = CustomMaxPool3d(
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.return_indices,
                module.ceil_mode
            )
            setattr(model, name, custom_mp)
            if isinstance(module.stride, tuple) and module.stride[0] != 1:
                dyn_depth_channel_stats //= module.stride[0]
            elif not isinstance(module.stride, tuple) and module.stride != 1:
                dyn_depth_channel_stats //= module.stride
        elif isinstance(module, nn.Module):
            replace_layers_arch_v2(module, dyn_depth_channel_stats)

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        pretrained_model = r3d_18(weights=weights)

        self.stem = pretrained_model.stem
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.avgpool = nn.AdaptiveAvgPool3d((16,1,1))
        num_ftrs = pretrained_model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        x= self.stem(x)
        x = self.layer_forward(self.layer1, x)
        x = self.layer_forward(self.layer2, x)
        x = self.layer_forward(self.layer3, x)
        x = self.layer_forward(self.layer4, x)
        x = self.avgpool(x)
        x = torch.cumsum(x.squeeze(-1).squeeze(-1), dim = 2)
        x = x/torch.arange(1, x.shape[2]+1).view(1,1,-1).to(device)
        x = x.permute(0,2,1)
        x = self.fc(x)
        return x

    
    def layer_forward(self, layer, x):
        for i, block in enumerate(layer):
            residual = x
            out = block.conv1(x)
            out = block.conv2(out)
            if block.downsample is not None:
                residual = block.downsample(x)
            out += residual
            out = block.relu(out)
            return out


if __name__ == "__main__":
    inputs = torch.rand(10, 3, 16, 112, 112)
    model = ModifiedResNet(num_classes=101)
    dyn_depth_channel_stats = 16
    replace_layers_arch_v1(model, depth_channel_stats=dyn_depth_channel_stats)
    model.layer4[0].conv1[1].depth_channel_stats = 2
    model.layer4[0].conv2[1].depth_channel_stats = 2
    model.layer4[0].downsample[1].depth_channel_stats = 2
    model.layer4[1].conv1[1].depth_channel_stats = 2
    model.layer4[1].conv2[1].depth_channel_stats = 2
    outputs = model(inputs)

    
