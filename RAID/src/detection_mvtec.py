import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import random
import torch
from src.utils import augment_image, dists2map, plot_ref_images



#构建memory bank
def run_anomaly_detection(
        model,
        cls,
        objects,
        data_root,
        n_ref_samples,
        nlist,
        object_anomalies,
        plots_dir,
        save_examples = False,
        masking = None,
        mask_ref_images = False,
        rotation = False,
        knn_metric = 'L2_normalized',
        knn_neighbors = 1,
        faiss_on_cpu = False,
        seed = 0,
        save_patch_dists = True,
        save_tiffs = False):
    """
    Main function to evaluate the anomaly detection performance of a given object/product.

    Parameters:
    - model: The backbone model for feature extraction (and, in case of DINOv2, masking).
    - object_name: The name of the object/product to evaluate.
    - data_root: The root directory of the dataset.
    - n_ref_samples: The number of reference samples to use for evaluation (k-shot). Set to -1 for full-shot setting.
    - object_anomalies: The anomaly types for each object/product.
    - plots_dir: The directory to save the example plots.
    - save_examples: Whether to save example images and plots. Default is True.
    - masking: Whether to apply DINOv2 to estimate the foreground mask (and discard background patches).
    - rotation: Whether to augment reference samples with rotation.
    - knn_metric: The metric to use for kNN search. Default is 'L2_normalized' (1 - cosine similarity)
    - knn_neighbors: The number of nearest neighbors to consider. Default is 1.
    - seed: The seed value for deterministic sampling in few-shot setting. Default is 0.
    - save_patch_dists: Whether to save the patch distances. Default is True. Required to eval detection.
    - save_tiffs: Whether to save the anomaly maps as TIFF files. Default is False. Required to eval segmentation.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    all_cls_prototypes = {}
    all_knn_index = {}
    gpu_id = torch.cuda.current_device()  # 当前 PyTorch 设备
    for object_name in objects:
        # add 'good' to the anomaly types
        object_anomalies[object_name].append('good')



        # Extract reference features
        features_ref = []
        images_ref = []
        masks_ref = []
        vis_backgroud = []
        #class kmeans
        cls_tokens = []

        img_ref_folder = f"{data_root}/{object_name}/train/good/"
        if n_ref_samples == -1:
            img_ref_samples = sorted(os.listdir(img_ref_folder))#用所有样本作为参考并且使用暴力搜索的方式查找模板，而且memory中存放的也是原始向量，而不是量化向量
        else:
            all_imgs = sorted(os.listdir(img_ref_folder))
            if object_name == "screw" or object_name == "wood" or object_name == "capsule" or object_name == "toothbrush":
                img_ref_samples = all_imgs
            else:
                img_ref_samples = random.sample(all_imgs, 80)#100

        if len(img_ref_samples) < n_ref_samples:
            print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")

        with torch.inference_mode():
            # start measuring time (feature extraction/memory bank set up)
            for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank", leave=False):
                # load reference image...
                img_ref = f"{img_ref_folder}{img_ref_n}"
                image_ref = cv2.cvtColor(cv2.imread(img_ref, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

                # augment reference image (if applicable)...
                if rotation[object_name]:
                    img_augmented = augment_image(image_ref)
                else:
                    img_augmented = [image_ref]
                for i in range(len(img_augmented)):
                    image_ref = img_augmented[i]
                    image_ref_tensor, grid_size1 = model.prepare_image(image_ref)
                    image_ref_tensor = image_ref_tensor.unsqueeze(0).cuda()
                    image_ref = torch.nn.functional.interpolate(image_ref_tensor, size=448, mode='bilinear',
                                                                align_corners=True)
                  
                    image_ref = image_ref.squeeze(0)
                    features_ref_i, cls_token = model.extract_features(image_ref)
                    cls_tokens.append(cls_token)
                 
                    # compute background mask and discard background patches
                    mask_ref = model.compute_background_mask(features_ref_i, grid_size1, threshold=10, masking_type=(mask_ref_images and masking[object_name]))
                    features_ref.append(features_ref_i[mask_ref])#mask中是true 和 false，是否需要mask，每个patch会有一个对应的mask来告诉特征patch是前景还是背景，只保留mask中是true的patch
                   

            features_ref = np.concatenate(features_ref, axis=0).astype('float32')
            cls_tokens = np.concatenate(cls_tokens, axis=0).astype('float32')
            kmeans = faiss.Kmeans(
                d=cls_token.shape[1],
                k=1, #每类聚类成3个中心
                niter=100,
                nredo=5,
                verbose=False,  # 可观察训练过程
                spherical=False,  # 是否归一化中心（通常否）
                seed=0,
                gpu=True  # ✅ 关键：启用 GPU
            )
            kmeans.train(cls_tokens)  # ✅ 只传数据，不要 index
            centers = kmeans.centroids
            all_cls_prototypes[object_name] = centers

            if knn_metric == "L2_normalized": #余弦相似度
                faiss.normalize_L2(features_ref) #会修改原始数组，需要注意

            if faiss_on_cpu:
                # similariy search on CPU
                knn_index = faiss.IndexFlatL2(features_ref.shape[1])
            else:
                # similariy search on GPU
                res = faiss.StandardGpuResources()
                res.setTempMemory(128 * 1024 * 1024)  # 限制为 128 MB（可调）

                quantizer = faiss.IndexFlatL2(features_ref.shape[1])  # 使用内积（IP）作为聚类依据，这里使用的是L2距离作为聚类依据，那么查询得到的就是L2距离
                index = faiss.IndexIVFFlat(quantizer, features_ref.shape[1], nlist)
                index.train(features_ref)

                #knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
                # knn_index = faiss.IndexFlatL2(features_ref.shape[1])
                # knn_index = faiss.index_cpu_to_gpu(res, int(model.device[-1]), knn_index)

            index.add(features_ref) #索引中存放的是归一化的特征

            knn_index = faiss.index_cpu_to_gpu(res, 0, index)  # 转移到 GPU 0
            all_knn_index[cls[object_name]] = knn_index
  
    return all_cls_prototypes, all_knn_index, grid_size1



