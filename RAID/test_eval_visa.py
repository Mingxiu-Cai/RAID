import os
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from src.utils import augment_image, dists2map, plot_ref_images
from src.post_eval import mean_top1p
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


def test_anomaly_detection(args, cls, model, net, masking, all_cls_prototypes, knn_metric, knn_index, knn_neighbors,
                           epoch):
    objects = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
               "pcb4", "pipe_fryum"]
    anomaly_scores = {}
    anomaly_maps = {}  # 类别 → [map1, map2, ...]
    labels_anomaly = {}  # 类别 → [label1, label2, ...] (0=good, 1=anomaly)
    groundtruth = {}
    predictions = []
    all_image_paths = []
    label = []
    net.eval()
    net.to(args.device)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    # Evaluate anomalies for each anomaly type (and "good")
    for root, dirs, files in os.walk(args.data_root):
        if os.path.basename(root) == 'test':
            object_name = Path(root).parent.name
            ground_truth_parent = Path(root).parent / 'ground_truth'
            # 第二步：在该 test 目录及其子目录中找所有图像
            for sub_root, sub_dirs, sub_files in os.walk(root):
                # 计算相对于 test 根目录的相对路径，用于定位 ground_truth 下的对应目录
                rel_path = Path(sub_root).relative_to(root)
                gt_sub_root = ground_truth_parent / rel_path  # 对应的 mask 文件所在目录

                for file in sub_files:
                    if Path(file).suffix.lower() in image_extensions:
                        stem = Path(file).stem
                        suffix = Path(file).suffix
                        mask_file = f"{stem}.png"  # 如 015.png -> 015_mask.png
                        mask_path = os.path.join(gt_sub_root, mask_file)

                        # 检查 mask 是否存在（对于 'good' 类别通常没有 mask）
                        if not Path(mask_path).exists():
                            mask_path = None  # 如果不存在，设置为 None 表示全0 mask

                        # all_image_paths.append(os.path.join(sub_root, file))
                        all_image_paths.append((os.path.join(sub_root, file), cls[object_name], mask_path))

   
    idx_to_class = {v: k for k, v in cls.items()}
    TARGET_H, TARGET_W = 256, 256
    with torch.no_grad():
        for image_test_path, cls_label, mask_path in all_image_paths:
            label_dir = Path(image_test_path).parts[-2]
            is_good = 1 if label_dir.lower() != 'good' else 0

            sims = []
            # Extract test features
            image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_tensor2, grid_size2 = model.prepare_image(image_test)
            image_tensor2 = image_tensor2.unsqueeze(0).cuda()
            image_ref = torch.nn.functional.interpolate(image_tensor2, size=448, mode='bilinear', align_corners=True)
            image_ref = image_ref.squeeze(0)

            features2, cls_token = model.extract_features(image_ref)

            feature_guidence = torch.from_numpy(features2).float().to(args.device)
            matching_logits = torch.zeros(features2.shape[0], knn_neighbors).to(args.device)  # 1024,100

            for name in objects:
                centers = all_cls_prototypes[name]  # [3, D]
                sim = cosine_similarity(cls_token, centers).max()
                sims.append(sim)

            pred_label = np.argmax(sims)
            labels_anomaly.setdefault(cls_label, []).append(is_good)

            if mask_path:
                mask = np.asarray(Image.open(mask_path).convert('L'))  # (H, W)
                mask = (mask > 128).astype(np.uint8)  # 二值化
                # resize 到目标尺寸
                mask = cv2.resize(mask, (TARGET_H, TARGET_W), interpolation=cv2.INTER_NEAREST)
                groundtruth.setdefault(cls_label, []).append(mask)
            else:
                mask_tensor = torch.zeros(256, 256).to(args.device)
                groundtruth.setdefault(cls_label, []).append(np.zeros((TARGET_H, TARGET_W), dtype=np.uint8))

            # Compute background mask
            if masking[idx_to_class[pred_label]]:
                mask2 = model.compute_background_mask(features2, grid_size2, threshold=10, masking_type=masking[
                    idx_to_class[pred_label]])  # 做特征的mask，背景特征不会再去查找模板了
            else:
                mask2 = np.ones(features2.shape[0], dtype=bool)
            feature_mask = features2[mask2]

            # 获取每个 query 最近的簇 ID
            faiss.normalize_L2(feature_mask)  # feature_mask是归一化后的特征
            _, cluster_ids = knn_index[pred_label].quantizer.search(feature_mask, k=1)
            cluster_ids = cluster_ids.flatten()  # [Q,]
            # 获取所有聚类中心（centroids）
            centroids = knn_index[pred_label].quantizer.reconstruct_n(0, args.nlist)  # [nlist, d]
            # centroids[i] 就是第 i 个簇的聚类中心
            # 获取每个 query 对应的聚类中心
            query_centroids = centroids[cluster_ids]  # [Q, d]
            # 将原始特征中的前景 patch 替换为聚类中心（重建）
            features2[mask2] = query_centroids  # 按顺序赋值，将mask2中true的位置（前景）替换为聚类中心，背景信息不变 1024, 384
            # print("features2",features2[idx].shape)

            # Compute distances to nearest neighbors in M
            if knn_metric == "L2":
                distances, match2to1 = knn_index.search(features2, k=knn_neighbors)
                if knn_neighbors > 1:
                    distances = distances.mean(axis=1)
                distances = np.sqrt(distances)

            elif knn_metric == "L2_normalized":
                if pred_label == "pcb1" or pred_label == "pcb2" or pred_label == "pcb3" or pred_label == "macaroni2":
                    knn_index[pred_label].nprobe = 20
                else:
                    knn_index[pred_label].nprobe = 20#5
                distances, indices = knn_index[pred_label].search(feature_mask,
                                                                  k=knn_neighbors)  # 返回的distance是squared L2 distance
                distances = distances / 2  # equivalent to cosine distance (1 - cosine similarity)
                # distances = 1.0 - distance

            matching_logits[mask2] = torch.from_numpy(distances).to(matching_logits.dtype).to(
                matching_logits.device)

            feature = torch.cat([feature_guidence, torch.from_numpy(features2).float().to(args.device)],
                                dim=-1)  # 在通道维度上拼接query及其聚类中心

           
            H = W = int(feature.shape[0] ** 0.5)  # 如果是正方形特征图
            assert H * W == feature.shape[0], "Q 必须能分解为 H × W"
            feature = feature.reshape(1, H, W, -1)  # [B, H, W, 2d:768]
            feature = feature.permute(0, 3, 1, 2)  # → [B, C, H, W]

            matching_logits = matching_logits.reshape(1, H, W, -1).permute(0, 3, 1, 2)  # torch.Size([1, 100, 32, 32])

            outputs, _, matching, _ = net(feature, matching_logits, epoch)  # 这个loss是负载均衡的loss, output: 1,1024(H*W)
            outputs = outputs + matching

            output_distances = F.interpolate(outputs.reshape(1, 1, H, W), size=args.resolution, mode='bilinear',
                                             align_corners=True)
            outputs[0][~mask2] = 0.0
            outputs = outputs.cpu().numpy()
            score = mean_top1p(outputs.flatten())
            anomaly_scores.setdefault(cls_label, []).append(score)

            output_distances = output_distances.squeeze()
            output_distances = output_distances.cpu().numpy()
            anomaly_map = dists2map(output_distances, [256, 256])

            anomaly_maps.setdefault(cls_label, []).append(anomaly_map)

        auroc_image_list = []
        auroc_pixel_list = []
        ap_pixel_list = []
        for class_id in sorted(idx_to_class.keys()):
            scores = np.array(anomaly_scores[class_id])
            labels = np.array(labels_anomaly[class_id])
            # 图像级指标
            auroc_img = roc_auc_score(labels, scores)
            auroc_image_list.append(auroc_img)
            print(f"Class {class_id} - "f"AUROC_img: {auroc_img:.3f}")

            # 像素级指标
            masks_flat = np.array(groundtruth[class_id]).ravel()
            preds_flat = np.array(anomaly_maps[class_id]).ravel()
            auroc_pix = roc_auc_score(masks_flat, preds_flat)
            ap_pix = average_precision_score(masks_flat, preds_flat)
            auroc_pixel_list.append(auroc_pix)
            ap_pixel_list.append(ap_pix)
           
            print(f"Class {class_id} - "
                  f"AUROC_pix: {auroc_pix:.3f}, AP_pix: {ap_pix:.3f}")

        # ✅ 计算平均指标
    mean_auroc_img = np.mean(auroc_image_list)
    mean_auroc_pix = np.mean(auroc_pixel_list)
    mean_ap_pix = np.mean(ap_pixel_list)
   

    print("\n" + "=" * 50)
    print("✅ Final Averaged Results:")
    print(f"Mean AUROC (Image):  {mean_auroc_img:.3f}")
    print(f"Mean AUROC (Pixel):   {mean_auroc_pix:.3f}")
    print(f"Mean AP     (Pixel):  {mean_ap_pix:.3f}")
    return mean_auroc_img



