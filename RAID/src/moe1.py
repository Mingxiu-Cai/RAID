import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .expert import Expert_layer1, Expert_layer2

from torch.optim.optimizer import Optimizer, required
from torch.optim import _functional

expert_num = 3  # MoE 模型中的专家数量（比如 4、8）


class NormalizedGD(Optimizer):  # 平衡不同专家的学习速度，对每个“专家”的梯度做归一化处理。
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, maximize=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NormalizedGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NormalizedGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            maximize = group['maximize']

            # 计算每个专家的总梯度范数（关键步骤！），这是“平衡不同专家学习速度”的核心逻辑！目标：防止某个专家因梯度过大主导训练。
            per_expert_num = int(len(group['params']) / expert_num)  # per_expert_num是每个专家拥有的参数张量数量。
            per_expert_norm = [0 for i in range(expert_num)]
            for i in range(expert_num):  # 假设有expert_num = 4个专家。
                for j in range(i * per_expert_num, (i + 1) * per_expert_num):  # 所有参数按顺序平均分配给每个专家（假设结构相同）
                    p = group['params'][j]
                    if p.grad is not None:
                        per_expert_norm[i] += p.grad.norm()  # 对第i个专家的所有参数，累加其梯度的L2范数 → 得到per_expert_norm[i]。

            # 遍历所有参数，归一化梯度并收集数据
            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    # Normalizing
                    if per_expert_norm[idx // per_expert_num] != 0:
                        p.grad /= per_expert_norm[
                            idx // per_expert_num]  # 将属于第 i = idx // per_expert_num 个专家的参数梯度除以该专家的总梯度范数。效果：所有专家的“整体梯度强度”被拉平。

                    # 收集需要更新的参数、梯度、动量缓存（用于 SGD with momentum）。
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # 使用 PyTorch 内部函数执行标准 SGD 更新。
            _functional.sgd(params_with_grad,
                            d_p_list,
                            momentum_buffer_list,
                            weight_decay=weight_decay,
                            momentum=momentum,
                            lr=lr,
                            dampening=dampening,
                            nesterov=nesterov,
                            maximize=maximize)

            # update momentum_buffers in state 把更新后的动量缓存写回 optimizer 的状态中，供下次使用。
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


# top 1 hard routing
# 从 tensor t 中选出最大值及其索引（沿最后一维），然后去掉最后维度。输出：values[B], index[B] —— 每个样本选择哪个专家 + 权重。
def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index  # 维度都是B


# hard routing according to probability
def choose1(t):
    index = t.multinomial(num_samples=1)
    values = torch.gather(t, 1, index)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def choose2(t):
    index = t.multinomial(num_samples=2)
    values = torch.gather(t, 1, index)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


# 安全的一键编码：避免 indexes 超出范围导致错误。
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]


class Router(nn.Module):
    # input_dim: 输入通道（3 表示 RGB），out_dim: 输出维度 = 专家数，strategy: 选择策略（未在内部使用），
    # patch_size, stride: 卷积核大小和步长，rtype: 路由器类型（卷积 or 全连接）
    def __init__(self, input_dim, out_dim, strategy='top1', patch_size=4, rtype='conv2d', stride=4):  # 4,4
        super(Router, self).__init__()
        if rtype == 'conv2d':
            self.conv1 = nn.Conv2d(input_dim, out_dim, patch_size, stride)
        elif rtype == 'linear':
            self.conv1 = nn.Linear(1728, out_dim)
        self.out_dim = out_dim
        self.strategy = strategy
        self.rtype = rtype
        # zero initialization， router的初始化应从0开始，加上噪声后可以进一步探索，保持负载均衡
        self.reset_parameters()  # 关键技巧：强制将权重和偏置初始化为零。目的是让初始 router 输出接近 0，再加噪声后接近均匀分布，促进探索。

    def reset_parameters(self):
        self.conv1.weight = torch.nn.Parameter(self.conv1.weight * 0)
        self.conv1.bias = torch.nn.Parameter(self.conv1.bias * 0)
        # nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        # nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        x = self.conv1(x)  # [B, expert_num, H', W']

        if self.rtype == 'conv2d':  # 在 H 和 W 维度求和 → 得到 [B, expert_num]，模拟全局池化。
            x = torch.mean(x, (2, 3))

        elif self.rtype == 'linear':
            x = torch.sum(x, 1)

        if self.training:  # 训练时加入均匀噪声，打破对称性，防止早期 collapse（所有样本都走同一个专家）。
            x = x + 0.01 * torch.rand(x.shape[0],
                                      self.out_dim).cuda()  # 添加随机噪声，促进探索，训练的时候在router中加噪声，防止早期就锁定某一个专家，提升多样性。
        else:
            x = x + torch.rand_like(x) * 0.05  # 保持轻微扰动
        return x  # 返回原始路由分数（后续由外部做 softmax）。


# --------------------------
# Router
# --------------------------
# class Router(nn.Module):
#     def __init__(self, input_dim, out_dim, rtype='conv2d', patch_size=4, stride=4):
#         super(Router, self).__init__()
#         if rtype == 'conv2d':
#             self.conv = nn.Conv2d(input_dim, out_dim, patch_size, stride)
#         elif rtype == 'linear':
#             self.conv = nn.Linear(input_dim, out_dim)
#         self.out_dim = out_dim
#         self.rtype = rtype
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.zeros_(self.conv.weight)
#         nn.init.zeros_(self.conv.bias)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.rtype == 'conv2d':
#             #x = x.sum(dim=2).sum(dim=2)  # global sum pool
#             x = x.mean(dim=(2, 3))  # (B, C_out, H, W) -> (B, C_out)
#
#         elif self.rtype == 'linear':
#             x = x.sum(dim=1)
#         if self.training:
#             x = x + torch.rand_like(x)
#         else:
#             x = x + torch.rand_like(x) * 0.05
#         return x

# --------------------------
# Top-2 Routing
# --------------------------
def top2_routing(select):
    values, indices = select.topk(2, dim=-1)  # top-2
    hard_mask = torch.zeros_like(select)
    hard_mask.scatter_(1, indices, 1.0)
    return values, indices, hard_mask


# --------------------------
# Nonlinear MoE
# --------------------------
class NonlinearMixtureRes(nn.Module):
    def __init__(self, expert_num, expert_num2, input_dim, out_dim, input_dim_matching, att_dim, strategy2=None,
                 strategy=None):
        super(NonlinearMixtureRes, self).__init__()
        self.router = Router(input_dim, expert_num)
        self.router2 = Router(out_dim + input_dim_matching + input_dim, expert_num2)
        self.models = nn.ModuleList([Expert_layer1(input_dim, out_dim) for _ in range(expert_num)])
        self.models2 = nn.ModuleList(
            [Expert_layer2(att_dim, out_dim + input_dim, input_dim_matching) for _ in range(expert_num2)])
        self.expert_num = expert_num
        self.expert_num2 = expert_num2

    # --------------------------
    # Forward 第一层
    # --------------------------
    def forward(self, x, matching_logits, epoch):
        B, C, H, W = x.shape
        select1 = F.softmax(self.router(x), dim=1)  # [B, expert_num]
        values, indices, hard_mask = top2_routing(select1)
        mask = hard_mask + (select1 - select1.detach())  # soft gradient

        # dispatch 给专家
        expert_inputs = []
        for i in range(self.expert_num):
            mask_i = mask[:, i].view(B, 1, 1, 1)
            expert_inputs.append(x * mask_i)
        expert_inputs = torch.stack(expert_inputs)  # [expert_num, B, C, H, W]

        # 第一层专家计算
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.models[i](expert_inputs[i]))  # [B, C+out_dim, H, W]
        outputs = torch.stack(outputs)  # [expert_num, B, C', H, W]

        # combine 第一层输出
        combine_tensor = select1.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, expert_num,1,1,1]
        combined_output = torch.einsum('be...,ebchw->bchw', combine_tensor, outputs)  # [B, C', H, W]

        # 负载均衡loss

        density_proxy = select1.mean(dim=0)
        density_actual = mask.mean(dim=0)
        load_balance_loss = (density_proxy * density_actual).mean() * self.expert_num2
        entropy_loss = -(select1 * torch.log(select1 + 1e-9)).sum(dim=1).mean()
        loss = load_balance_loss + 0.01 * entropy_loss

        # 第二层 MoE
        output2, matching, loss2 = self.moe_dense_layer2(combined_output, matching_logits, epoch)
        loss = loss + loss2

        return output2, loss, matching, loss2

    def moe_dense_layer2(self, x, matching_logits, epoch):
        if epoch < 30:
            output1, matching = self.models2[0](x, matching_logits)
            output2, matching = self.models2[1](x, matching_logits)
            output3, matching = self.models2[2](x, matching_logits)
            output = (output1 + output2 + output3) / 3
            load_balance_loss = 0
        else:
            select = self.router2(torch.cat([x, matching_logits], dim=1))
            B, E = select.shape
            tau = 1  
            select = F.softmax(select / tau, dim=1)
            select = select + 1e-2
            select = select / select.sum(dim=1, keepdim=True)
            importance = select.mean(dim=0)
            load_balance_loss = (importance * importance).sum() * E
            expert_outputs = []  # 存每个专家的输出: [B, C', H', W']

            for i in range(self.expert_num2):
                out_i, matching = self.models2[i](x, matching_logits)  # 每个专家处理完整 batch [B, C', H', W']
                expert_outputs.append(out_i)

            # Stack 成 [3, B, C', H', W']
            expert_outputs = torch.stack(expert_outputs)  # shape: [expert_num, B, C', H', W']
            output = torch.einsum('be,ebh->bh', select, expert_outputs)

        return output, matching, load_balance_loss

    # --------------------------
    # 监控专家选择
    # --------------------------
    def return_select(self, x):
        select = self.router1(x)
        _, indices, _ = top2_routing(F.softmax(select, dim=1))
        return indices  # [B, 2]


