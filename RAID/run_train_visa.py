import argparse
import os
import cv2
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"
from argparse import ArgumentParser, Action
import yaml
from torchvision import transforms
from tqdm import trange
from tqdm import tqdm
import torch
from torch import optim
from src.moe1 import NonlinearMixtureRes
import csv
import numpy as np
import faiss
from src.loss import FocalLoss, SSIM, BinaryFocalLoss
from src.post_eval import mean_top1p
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from tensorboard_visualizer import TensorboardVisualizer
from test_eval_visa import test_anomaly_detection
from sklearn.metrics import roc_auc_score

from src.utils import get_dataset_info
from src.detection_visa import run_anomaly_detection
from src.post_eval import eval_finished_run
from src.visualize import create_sample_plots
from src.backbones import get_model
from dataset.dataset_multiclass import MVTecDataset
from src.utils import augment_image, dists2map, plot_ref_images
from torch.utils.data import DataLoader, Subset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(pred, target, smooth=1e-6):
    """
    pred: 模型输出，经过 sigmoid 后的概率图，shape: (B, 1, H, W) 或 (B, H, W)
    target: 真实 mask，0/1 的 float tensor，shape: (B, 1, H, W)
    smooth: 防止除零的小数
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"

    pred = pred.contiguous().view(-1)  # 展平
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice  # 越小越好


def print_grad_norms(model, name="Model"):
    total_norm = 0.
    param_count = 0

    for n, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:  # 检查是否需要计算梯度
            param_norm = p.grad.data.norm(2).item()
            print(f"{name} | {n} (requires_grad={p.requires_grad}): {param_norm}")
            print(f"{n}: grad mean={p.grad.mean().item()}, std={p.grad.std().item()}")
            total_norm += param_norm ** 2
            param_count += 1
        elif not p.requires_grad:  # 如果参数不需要计算梯度，则打印出来
            print(f"{name} | {n} is frozen (requires_grad={p.requires_grad})")
    total_norm = total_norm ** 0.5
    print(f"{name} | Gradient Norm (Total): {total_norm} | Trained Params: {param_count}")


class IntListAction(Action):
    """
    Define a custom action to always return a list.
    This allows --shots 1 to be treated as a list of one element [1].
    """

    def __call__(self, namespace, values):
        if not isinstance(values, list):
            values = [values]
        setattr(namespace, self.dest, values)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14",
                        help="Name of the backbone model. Choose from ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'vit_b_16'].")
    parser.add_argument("--data_root", type=str, default="/home/CMX/datas/VisA",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--expert_num1", default=3)
    parser.add_argument("--expert_num2", default=3)
    parser.add_argument("--strategy1", default="top1")
    parser.add_argument("--strategy2", default="top1")
    parser.add_argument("--nlist", default=50)
    parser.add_argument("--att_dim", default=1024, help="Attention map dimention: H*W")
    parser.add_argument("--preprocess", type=str, default="agnostic",
                        help="Preprocessing method. Choose from ['agnostic', 'informed', 'masking_only'].")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction,
                        help="Use GPU for FAISS kNN search. (Conda install faiss-gpu recommended, does usually not work with pip install.)")
    parser.add_argument("--shots", nargs='+', type=int, default=[1],  # action=IntListAction,
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mask_ref_images", default=False)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument('--save_examples', default=True, action=argparse.BooleanOptionalAction,
                        help="Save example plots.")
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction,
                        help="Evaluate anomaly detection performance.")
    parser.add_argument("--eval_segm", default=False, action=argparse.BooleanOptionalAction,
                        help="Evaluate anomaly segmentation performance.")
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--warmup_iters", type=int, default=25,
                        help="Number of warmup iterations, relevant when benchmarking inference time.")

    parser.add_argument("--tag", help="Optional tag for the saving directory.")

    args = parser.parse_args()
    return args


def train_anomaly_detection(
        args,
        model,
        grid_size2,
        knn_index,
        knn_neighbors=1,
        masking=None,
        knn_metric='L2_normalized',

):
    # 加载数据
    train_dataset = MVTecDataset(
        instance_data_root=args.data_root,
        class_name='',
        resize=args.resolution,
        img_size=args.resolution,
        train=True
    )
    N = 100
    subset_dataset = Subset(train_dataset, range(N))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=3,
    )

    net = NonlinearMixtureRes(args.expert_num1, args.expert_num2, 768, 384, knn_neighbors, args.att_dim, args.strategy2,
                              strategy=args.strategy1).to(args.device)

    # net = NonlinearMixtureRes(args.expert_num1, args.expert_num2, 768, 384, knn_neighbors, args.att_dim).to(args.device)

    # optimizer = NormalizedGD(net.models.parameters(), lr=1e-8, momentum=0.9,
    #                          weight_decay=5e-4)  # 专家网络（Expert Networks）的优化器，专家模型用的是resnet或者MobileNetV2
    # # 使用 NormalizedGD 可以：让每个专家的更新步长更均衡，避免“强者恒强，弱者恒弱”。这是一种 专家负载均衡（Load Balancing） 的隐式手段。防止某些专家更新过猛，促进负载均衡
    # # optimizer = torch.optim.Adam([
    # #                                   {"params": net.models.parameters(), "lr": 1e-2}])
    # optimizer2 = optim.SGD(net.router.parameters(), lr=1e-8,
    #                        momentum=0.9, weight_decay=5e-4)  # 路由网络的优化器
    # # optimizer2 = torch.optim.Adam([
    # #     {"params": net.router.parameters(), "lr": 1e-6}])
    #
    #
    # optimizer_model2 = NormalizedGD(net.models2.parameters(), lr=1e-6, momentum=0.9,
    #                                 weight_decay=5e-4)  # 专家网络（Expert Networks）的优化器，专家模型用的是resnet或者MobileNetV2
    # # 使用 NormalizedGD 可以：让每个专家的更新步长更均衡，避免“强者恒强，弱者恒弱”。这是一种 专家负载均衡（Load Balancing） 的隐式手段。防止某些专家更新过猛，促进负载均衡
    # # optimizer_model2 = torch.optim.Adam([
    # #     {"params": net.models2.parameters(), "lr": 1e-2}])
    # optimizer_router2 = optim.SGD(net.router2.parameters(), lr=1e-6,
    #                               momentum=0.9, weight_decay=5e-4)  # 路由网络的优化器
    # # optimizer_router2 = torch.optim.Adam([
    # #                                   {"params": net.router2.parameters(), "lr": 1e-10}])
    #
    # optimizers = [optimizer, optimizer2, optimizer_model2, optimizer_router2]  # 专家需要快学，路由器需要稳学，因此使用了不同的学习率
    optims = torch.optim.Adam([
        {"params": net.parameters(), "lr": 1e-4}])
    criterion = BinaryFocalLoss().to(args.device)
    loss_ssim = SSIM().to(args.device)
    bceloss = nn.BCELoss()
    bceloss_logits = nn.BCEWithLogitsLoss()

    # schedulers = [
    #     torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=200),  # optimizer
    #     #torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[2], T_max=200),  # optimizer_model2
    # ]
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # visualizer = TensorboardVisualizer(log_dir="/home/CMX/anomaly-our/retrieval-Generate/")

    n_iter = 0

    for epoch in range(args.epochs):
        print("Epoch: " + str(epoch))
        net.train()
        anomaly_scores = {}
        anomaly_maps = {}
        labels_anomaly = {}

        for step, batch in enumerate(train_dataloader):
            optims.zero_grad()

            instance_images = batch["anomaly_images"].to(args.device)
            objects = batch["class_name"]  # ['bottle', 'cable', 'pill', ...]，是一个列表
            groundtruth = batch["anomaly_masks"].to(args.device)  # [B, 1, H, W]

            images = torch.nn.functional.interpolate(instance_images, size=448, mode='bilinear', align_corners=True)
            features2, _ = model.extract_features(images)  # B,H*W,C
            feature_guidence = torch.from_numpy(features2).float().to(args.device)
            matching_logits = torch.zeros(features2.shape[0], features2.shape[1], knn_neighbors).to(args.device)
            matching_logits += 0.001
            # Compute background mask，用作最后计算结果的过滤
            idx = 0

            for obj in objects:
                if masking[obj]:
                    mask2 = model.compute_background_mask(features2[idx], grid_size2, threshold=10,
                                                          masking_type=masking[obj])  # 做特征的mask，背景特征不会再去查找模板了
                else:
                    mask2 = np.ones(features2[idx].shape[0], dtype=bool)  # 1024
                # Discard irrelevant features
                _, D = features2[idx][mask2].shape

                feature_mask = features2[idx][mask2].reshape(-1, D)
                # 获取每个 query 最近的簇 ID
                faiss.normalize_L2(feature_mask)  # feature_mask是归一化后的特征
                _, cluster_ids = knn_index[cls[obj]].quantizer.search(feature_mask, k=1)
                cluster_ids = cluster_ids.flatten()  # [Q,]
                # 获取所有聚类中心（centroids）
                centroids = knn_index[cls[obj]].quantizer.reconstruct_n(0, args.nlist)  # [nlist, d]
                # centroids[i] 就是第 i 个簇的聚类中心
                # 获取每个 query 对应的聚类中心
                query_centroids = centroids[cluster_ids]  # [Q, d]
                # 将原始特征中的前景 patch 替换为聚类中心（重建）
                features2[idx][mask2] = query_centroids  # 按顺序赋值，将mask2中true的位置（前景）替换为聚类中心，背景信息不变 1024, 384
                # print("features2",features2[idx].shape)

                # Compute distances to nearest neighbors in M
                if knn_metric == "L2":
                    distances, match2to1 = knn_index.search(features2, k=knn_neighbors)
                    if knn_neighbors > 1:
                        distances = distances.mean(axis=1)
                    distances = np.sqrt(distances)

                elif knn_metric == "L2_normalized":
                    if obj == "pcb1" or obj == "pcb2" or obj == "pcb3" or obj == "macaroni2":
                        knn_index[cls[obj]].nprobe = 20
                    else:
                         knn_index[cls[obj]].nprobe = 5
                    distances, indices = knn_index[cls[obj]].search(feature_mask,
                                                                    k=knn_neighbors)  # 返回的distance是squared L2 distance,输入应该是object name
                    distances = distances / 2  # equivalent to cosine distance (1 - cosine similarity), 前景的余弦相似度 H*W,knn_neighbors
                matching_logits[idx][mask2] = torch.from_numpy(distances).to(matching_logits.dtype).to(
                    matching_logits.device)
                idx += 1

            feature = torch.cat([feature_guidence, torch.from_numpy(features2).float().to(args.device)],
                                dim=-1)  # 在通道维度上拼接query及其聚类中心
            H = W = int(feature.shape[1] ** 0.5)  # 如果是正方形特征图
            assert H * W == feature.shape[1], "Q 必须能分解为 H × W"
            feature = feature.reshape(feature.shape[0], H, W, -1)  # [B, H, W, 2d:768]
            feature = feature.permute(0, 3, 1, 2)  # → [B, C, H, W]

            matching_logits = matching_logits.reshape(feature.shape[0], H, W, -1).permute(0, 3, 1, 2)

            outputs, loss_balance, _, _ = net(feature, matching_logits, epoch)  # 这个loss是负载均衡的loss

            ground = groundtruth.reshape(-1, 1, args.resolution, args.resolution)

            outputs = outputs.reshape(-1, 1, H, W)
            outputs = F.interpolate(outputs, size=(args.resolution, args.resolution), mode='bilinear',
                                          align_corners=True)
            loss_focal = criterion(outputs, ground)
            loss = loss_focal + 0.005 * loss_balance
            loss.backward()
            optims.step()
            n_iter += 1

        checkpoint_dir = f"checkpoints/{args.dataset}/{args.resolution}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch{epoch}_seed{seed}_150_score.pth"

        if epoch >= 30:
            result = test_anomaly_detection(args, cls, model, net, masking, all_cls_prototypes, knn_metric, knn_index,
                                            knn_neighbors, 100)
            # 保存 checkpoint
            torch.save({
                'model_state_dict': net.state_dict(),
            }, checkpoint_path)

            print(f"Saved checkpoint at {checkpoint_path}")



if __name__ == "__main__":

    args = parse_args()

    print(f"Requested to run {len(args.shots)} (different) shot(s):", args.shots)
    print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    cls, idx_to_cls, objects, object_anomalies, masking_default, rotation_default = get_dataset_info(args.dataset,
                                                                                                     args.preprocess)

    # set CUDA device
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    model = get_model(args.model_name, 'cuda', smaller_edge_size=args.resolution)

    if not args.model_name.startswith("dinov2"):
        masking_default = {o: False for o in objects}
        print("Caution: Only DINOv2 supports 0-shot masking (for now)!")

    if args.just_seed != None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)

    for shot in list(args.shots):
        save_examples = args.save_examples

        results_dir = f"results_{args.dataset}/{args.model_name}_{args.resolution}/{shot}-shot_preprocess={args.preprocess}"

        if args.tag != None:
            results_dir += "_" + args.tag
        plots_dir = results_dir
        os.makedirs(f"{results_dir}", exist_ok=True)

        # save preprocessing setups (masking and rotation) to file
        with open(f"{results_dir}/preprocess.yaml", "w") as f:
            yaml.dump({"masking": masking_default, "rotation": rotation_default}, f)

        # save arguments to file
        with open(f"{results_dir}/args.yaml", "w") as f:
            yaml.dump(vars(args), f)

        if args.faiss_on_cpu:
            print("Warning: Running similarity search on CPU. Consider using faiss-gpu for faster inference.")

        print("Results will be saved to", results_dir)

        for seed in seeds:
            set_seed(seed)
            print(f"=========== Shot = {shot}, Seed = {seed} ===========")

            if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                print(f"Results for shot {shot}, seed {seed} already exist. Skipping.")
                continue
            else:
                timeit_file = results_dir + "/time_measurements.csv"
                with open(timeit_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])

                    all_cls_prototypes, knn_index, grid_size2 = run_anomaly_detection(
                        model,
                        cls,
                        objects,
                        data_root=args.data_root,
                        n_ref_samples=shot,
                        nlist=args.nlist,
                        object_anomalies=object_anomalies,
                        plots_dir=plots_dir,
                        save_examples=save_examples,
                        knn_metric=args.knn_metric,
                        knn_neighbors=args.k_neighbors,
                        faiss_on_cpu=args.faiss_on_cpu,
                        masking=masking_default,
                        mask_ref_images=args.mask_ref_images,
                        rotation=rotation_default,
                        seed=2,
                        save_patch_dists=args.eval_clf,  # save patch distances for detection evaluation
                        save_tiffs=args.eval_segm)  # save anomaly maps as tiffs for segmentation evaluation

                    train_anomaly_detection(
                        args,
                        model,
                        grid_size2=grid_size2,
                        knn_index=knn_index,
                        knn_neighbors=150,
                        masking=masking_default,
                        knn_metric='L2_normalized',
                    )




    print("Finished and evaluated all runs!")