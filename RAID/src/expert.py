import torch
import torch.nn as nn


class Expert_layer1(nn.Module):
    def __init__(self, input_dim, out_dim):  # 4,4
        super(Expert_layer1, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = torch.cat([res, x], dim=1)  # B,input_dim+out_dim,H,W

        return x


# class Expert_layer2(nn.Module):
#     def __init__(self, dim, out_dim_feature, input_dim_matching):  # 4,4
#         super(Expert_layer2, self).__init__()
#
#         self.att_dim = dim
#         self.out_dim_feature = out_dim_feature
#         self.input_dim_matching = input_dim_matching
#
#         self.q = nn.Linear(self.input_dim_matching, self.input_dim_matching)
#         # self.k = nn.Linear(self.out_dim_feature, self.out_dim_feature // 2)
#         self.k = nn.Linear(self.input_dim_matching, self.input_dim_matching)
#         self.v = nn.Linear(self.input_dim_matching, self.input_dim_matching)
#         # self.att_reshape = nn.Linear(self.att_dim, self.input_dim_matching) #attention 的reshape，从H*W变为K维度
#
#         self.norm_x1 = nn.LayerNorm(self.input_dim_matching)
#         self.norm_x2 = nn.LayerNorm(self.input_dim_matching)  # K
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.input_dim_matching, out_channels=self.input_dim_matching // 2, kernel_size=3,
#                       padding=1),
#             nn.BatchNorm2d(self.input_dim_matching // 2),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=self.input_dim_matching // 2, out_channels=self.input_dim_matching, kernel_size=1,
#                       padding=0),
#             nn.BatchNorm2d(self.input_dim_matching),
#             #nn.Sigmoid()
#             # nn.ReLU(inplace=True),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.out_dim_feature, out_channels=self.out_dim_feature // 2, kernel_size=3,
#                       padding=1),
#             nn.BatchNorm2d(self.out_dim_feature // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=self.out_dim_feature // 2, out_channels=self.input_dim_matching, kernel_size=3,
#                       padding=1),
#             nn.BatchNorm2d(self.input_dim_matching),
#             nn.ReLU(inplace=True),
#         )
#         # self.conv4 = nn.Sequential(
#         #     nn.Linear(self.input_dim_matching, self.input_dim_matching // 2),
#         #     nn.LayerNorm(self.input_dim_matching // 2),  # 加归一化
#         #     nn.GELU(),
#         #     nn.Linear(self.input_dim_matching // 2, 1)
#         # )
#         self.conv4 = nn.Linear(self.input_dim_matching, 1)
#
#         self.att_reshape = nn.Sequential(
#             nn.Linear(self.att_dim, self.att_dim // 2),
#             nn.GELU(),
#             nn.Linear(self.att_dim // 2, self.input_dim_matching),
#             #nn.ReLU()
#         )
#         self.norm_fuse = nn.LayerNorm(self.input_dim_matching)
#         self.norm_confidence = nn.LayerNorm(self.input_dim_matching)
#         self.cross_cbam = CrossCBAM(x1_channels=self.input_dim_matching, x2_channels=self.out_dim_feature, reduction=16)
#
#         self.reset_parameter()  # ? 添加统一初始化
#
#
#
#     def reset_parameter(self):
#         # --- 对 Linear 层做 Xavier 初始化 ---
#         nn.init.xavier_uniform_(self.q.weight)
#         nn.init.zeros_(self.q.bias)
#         nn.init.xavier_uniform_(self.k.weight)
#         nn.init.zeros_(self.k.bias)
#
#         for m in self.att_reshape:
#             if isinstance(m, nn.Linear):
#
#                 nn.init.xavier_uniform_(m.weight, gain=1e-2)
#                 nn.init.zeros_(m.bias)
#
#
#         for m in [self.conv1, self.conv2]:
#             for layer in m:
#                 if isinstance(layer, nn.Conv2d):
#                     nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#                     nn.init.zeros_(layer.bias)
#                 elif isinstance(layer, nn.BatchNorm2d):
#                     nn.init.ones_(layer.weight)
#                     nn.init.zeros_(layer.bias)
#         last_linear = self.conv4
#         nn.init.xavier_uniform_(last_linear.weight, gain=1.0)  # 适合 sigmoid/tanh
#         nn.init.constant_(last_linear.bias, 0.0)
#
#     # def forward1(self, x1, x2):
#     #     #x1:B,C,H,W, x2:B,K,H,W
#     #     matching = x2
#     #     x1 = x1.reshape(x1.shape[0], x1.shape[1], -1).permute(0,2,1) #B,H*W,C
#     #     x2 = x2.reshape(x2.shape[0], x2.shape[1], -1).permute(0, 2, 1)  # B,H*W,K
#     #
#     #     q = self.q(x1)
#     #     k = self.k(x2)
#     #     attention = (q @ k.transpose(-2, -1))
#     #     attention = self.att_reshape(attention)
#     #     attention = torch.softmax(attention, dim=-1) #B,H*W,K [16, 1024, 10]
#     #
#     #
#     #     matching = self.conv1(matching)
#     #     confidence = self.conv2(matching)
#     #     confidence = confidence.reshape(confidence.shape[0], confidence.shape[1], -1).permute(0,2,1) #B,H*W,K [16, 1024, 10]
#     #
#     #     weight = torch.softmax(attention * confidence, dim=-1)
#     #     x = torch.sum(x2 * weight, dim=-1) #B, H*W
#     #
#     #     return x
#
#     def forward(self, x1, x2):
#
#
#
#         # x2:B,C,H,W
#         cross_cbam1 = self.cross_cbam.cuda()
#         # 前向
#         # x1 = torch.randn(2, 64, 32, 32)  # 主特征
#         # x2 = torch.randn(2, 10, 32, 32)  # 匹配分数或辅助特征
#         x_fused = cross_cbam1(x2, x1)  # [2, 64, 32, 32]
#
#         print('cbam min/max/mean:', x_fused.min().item(), x_fused.max().item(), x_fused.mean().item())
#
#         matching = x2
#         # x1 = self.conv3(x1)
#         B, C, H, W = x2.shape
#
#
#
#         #x1 = x1.reshape(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
#         x_fused = x_fused.reshape(B, C, H * W).permute(0, 2, 1)
#         x2 = x2.reshape(B, x2.shape[1], -1).permute(0, 2, 1)  # [B, HW, K]
#         # x2_norm = self.norm_x2(x2)
#         # x1 = self.norm_x1(x1)
#         # #x1 = self.norm_x1(x1)  # [B, HW, C]
#         # #x2 = self.norm_x2(x2)
#         # # print("x1 mean/std:", x1.mean().item(), x1.std().item())
#         # # print("x2 mean/std:", x2.mean().item(), x2.std().item())
#         #
#         # q = self.q(x1)
#         # k = self.k(x2)
#         # # v = self.v(x2)
#         # attention = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
#         # print('attention_origin min/max/mean:', attention.min().item(), attention.max().item(), attention.mean().item())
#         # #attention = self.att_reshape(attention)
#         # attention = torch.softmax(attention, dim=-1)
#         # #attention = torch.sigmoid(attention)
#         # print('attention min/max/mean:', attention.min().item(), attention.max().item(), attention.mean().item())
#         #
#         # # x2_weight1 = attention * x2
#         # # print('attention1 min/max/mean:', x2_weight1.min().item(), x2_weight1.max().item())
#         #
#         #
#         # # attention = attention @ v
#         #
#         # # attention = self.att_reshape(attention)
#         # # attention = attention / (attention.mean(dim=(1, 2), keepdim=True) + 1e-6)
#         #
#         # # print("attention", attention)
#
#         matching = self.conv1(matching)
#         confidence = self.conv2(matching)  # no sigmoid
#
#         # confidence = confidence / (confidence.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
#
#         confidence = confidence.reshape(B, confidence.shape[1], -1).permute(0, 2, 1)
#         confidence = torch.softmax(confidence, dim=-1)
#         #confidence = self.norm_confidence(confidence)
#
#         # print("confidence", confidence)
#         print('confidence min/max/mean:', confidence.min().item(), confidence.max().item(), confidence.mean().item())
#
#         x2_weight2 = confidence * x2 #+ x2_weight1
#         #x2_weight2 = torch.sigmoid(x2_weight2)
#         print('weight min/max/mean:', x2_weight2.min().item(), x2_weight2.max().item(), x2_weight2.mean().item())
#
#         # weight = (attention * confidence)
#         # weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-6)
#         #x = attention @ x2_weight2
#         #print('weight_x min/max/mean:', x.min().item(), x.max().item(), x.mean().item())
#
#         x = x2_weight2 + x_fused + x2
#         print('x min/max/mean:', x.min().item(), x.max().item(), x.mean().item())
#
#         #x = self.norm_fuse(x)
#         x = self.conv4(x).squeeze(-1)
#         # x = torch.sum(x2 * weight, dim=-1)
#         # x = torch.mean(x2_weight2, dim=-1)
#
#         #+ x2.mean(dim=-1) * 0.1  # residual path
#
#         print('logit min/max/mean:', x.min().item(), x.max().item())
#         x = torch.sigmoid(x)
#         #x = x2.min(dim=-1)[0]
#         print('pred min/max/mean:', x.min().item(), x.max().item())
#         # print("x",x)
#         # print("logits mean:", x.mean().item(), "std:", x.std().item())
#
#         return x, 1



class Expert_layer2(nn.Module):
    """
    x1: guided feature      [B, Fx1, H, W]
    x2: to-be-filtered      [B, C2,  H, W]

    """
    def __init__(
        self,
        dim: int,
        out_dim_feature: int,
        input_dim_matching: int,
        num_groups: int = 5,
        gate_scale: float = 0.1, #0.1
        use_sigmoid: bool = True,
        head_mode: str = 'conv3x3',      # 'linear' | 'conv1x1' | 'conv3x3'
        head_gn_groups: int = 5
    ):
        super().__init__()
        assert head_mode in {'linear', 'conv1x1', 'conv3x3'}
        self.att_dim = dim
        self.Fx1    = out_dim_feature          # x1 channels
        self.K      = input_dim_matching       # match/attention channels
        self.C2     = out_dim_feature
        self.gate_scale = gate_scale
        self.use_sigmoid = use_sigmoid
        self.head_mode = head_mode

        # x1 -> K
        self.proj_x1 = nn.Sequential(
            nn.Conv2d(self.Fx1, self.K, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, self.K),
            nn.SiLU(),
        )

        # x2 -> K
        # if self.C2 != self.K:
        #     self.to_K = nn.Conv2d(self.C2, self.K, kernel_size=1, bias=False)
        # else:
        #     self.to_K = nn.Identity()


        self.proj_x2_id = nn.Sequential(
            nn.Conv2d(self.K, self.K, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, self.K),
        )

        # q/k/v
        self.q = nn.Linear(self.K, self.K, bias=False)
        self.k = nn.Linear(self.K, self.K, bias=False)
        self.v = nn.Linear(self.K, self.K, bias=False)


        self.gate = nn.Sequential(
            nn.Conv2d(self.K, self.K // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, self.K // 2),
            nn.SiLU(),
            nn.Conv2d(self.K // 2, self.K, kernel_size=1, bias=True),
        )


        self.ln_x1  = nn.LayerNorm(self.K)
        self.ln_x2  = nn.LayerNorm(self.K)
        self.ln_out = nn.LayerNorm(self.K)

        # Head
        if self.head_mode == 'linear':

            self.head_linear = nn.Linear(self.K, 1)
            nn.init.zeros_(self.head_linear.bias)

        elif self.head_mode == 'conv1x1':

            self.head_conv1x1 = nn.Conv2d(self.K, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.head_conv1x1.bias)

        else:  # 'conv3x3'
            assert self.K % head_gn_groups == 0, \
                f"K ({self.K})  head_gn_groups ({head_gn_groups}) "
            self.head_conv = nn.Sequential(
                nn.Conv2d(self.K, self.K, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(head_gn_groups, self.K),
                nn.SiLU(),
                nn.Conv2d(self.K, 1, kernel_size=1, bias=True),
            )

            nn.init.zeros_(self.head_conv[-1].bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        x1: [B, Fx1, H, W]
        x2: [B, C2,  H, W]
        return:
            pred or logits: [B, H, W]
        """
        B, _, H, W = x1.shape
        x2k = x2                       # [B, K, H, W]

        # x1 → K；x2
        x1p   = self.proj_x1(x1)                   # [B, K, H, W]
        x2_id = self.proj_x2_id(x2k)               # [B, K, H, W]


        x1s = x1p.flatten(2).transpose(1, 2)       # [B, HW, K]
        x2s = x2k.flatten(2).transpose(1, 2)       # [B, HW, K]
        x1s = self.ln_x1(x1s)
        x2s = self.ln_x2(x2s)

        # q/k/v
        q = self.q(x1s)                            # [B, HW, K]
        k = self.k(x2s)                            # [B, HW, K]
        v = self.v(x2s)                            # [B, HW, K]

        #
        g = self.gate(x2k)                         # [B, K, H, W]
        g = torch.sigmoid(g).flatten(2).transpose(1, 2)  # [B, HW, K]
        v = v * (1.0 + self.gate_scale * g)

        #
        attn = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)  # [B, HW, HW]
        attn = attn.softmax(dim=-1)
        y = attn @ v                                            # [B, HW, K]

        #
        x2_id_seq = x2_id.flatten(2).transpose(1, 2)            # [B, HW, K]
        y = self.ln_out(y + x2_id_seq)                          # [B, HW, K]

        # Head
        if self.head_mode == 'linear':
            logits = self.head_linear(y).squeeze(-1)            # [B, HW]
            logits = logits.view(B, H*W)                       # [B, H, W]
        else:
            y_map = y.transpose(1, 2).reshape(B, self.K, H, W)  # [B, K, H, W]
            if self.head_mode == 'conv1x1':
                logits = self.head_conv1x1(y_map).squeeze(1)    # [B, H, W]
            else:  # 'conv3x3'
                logits = self.head_conv(y_map).squeeze(1)       # [B, H, W]
            logits = logits.view(B, H*W)

        if self.use_sigmoid:
            x = torch.min(x2, dim=1)[0]
            return torch.sigmoid(logits), x.reshape(B, H*W) 
        else:
            return logits, 0






class CrossCBAM(nn.Module):
    def __init__(self, x1_channels, x2_channels, reduction=16):
        super(CrossCBAM, self).__init__()
        self.x1_channels = x1_channels
        self.x2_channels = x2_channels

        # 1. Cross Channel Attention: 使用 x2 的全局信息生成 x1 的通道权重
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(x2_channels, x1_channels // reduction),
            nn.ReLU(),
            nn.Linear(x1_channels // reduction, x1_channels),
            nn.Sigmoid()
        )

        # 2. Cross Spatial Attention: 使用 x1 和 x2 的拼接生成空间权重
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(x1_channels + x2_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, C1, H, W = x1.shape
        _, C2, _, _ = x2.shape

        # --- 1. Cross Channel Attention ---
        # 用 x2 的全局信息生成 x1 的通道注意力
        x2_pooled = self.channel_pool(x2).view(B, C2)  # [B, C2]
        channel_weights = self.channel_mlp(x2_pooled).view(B, C1, 1, 1)  # [B, C1, 1, 1]
        x1_attended = x1 * channel_weights  # [B, C1, H, W]

        # --- 2. Cross Spatial Attention ---
        # 拼接 x1 和 x2，在空间维度上生成注意力
        combined = torch.cat([x1_attended, x2], dim=1)  # [B, C1+C2, H, W]
        spatial_weights = self.spatial_conv(combined)  # [B, 1, H, W]
        x_out = x1_attended * spatial_weights  # [B, C1, H, W]

        return x_out

    