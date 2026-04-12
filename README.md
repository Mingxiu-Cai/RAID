# RAID: Retrieval-Augmented Anomaly Detection
### [Project Page](https://github.com/Mingxiu-Cai/RAID) | [Paper](https://arxiv.org/abs/2602.19611)

The official repository for [**RAID: Retrieval-Augmented Anomaly Detection**](https://arxiv.org/abs/2602.19611), accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026).

![Image](https://github.com/Mingxiu-Cai/RAID/blob/main/images/pipeline.png)
We propose RAID, a retrieval-augmented UAD framework designed for noise-resilient anomaly detection and localization. Unlike standard RAG that enriches context or knowledge, we focus on using retrieved normal samples to guide noise suppression in anomaly map generation. RAID retrieves class-, semantic-, and instance-level representations from a hierarchical vector database, forming a coarse-to-fine pipeline. A matching cost volume correlates the input with retrieved exemplars, followed by a guided Mixture-of-Experts (MoE) network that leverages the retrieved samples to adaptively suppress matching noise
and produce fine-grained anomaly maps.

## Installation
Create a new conda environment and install the required packages using the environment.yaml file


```bash
conda env create -f environment.yaml
``` 


##  📊 Dataset

Download and prepare the datasets [MVTec-AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff) from their official sources.

## 🚀 Running
```bash
python run_train_visa.py --dataset VisA --num_seeds 1 --preprocess masking_only
``` 
or
```bash
python run_train_mvtec.py --dataset MVTec --num_seeds 1 --preprocess masking_only
``` 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📰 News

- **Nov 13, 2025** — Paper submitted to CVPR 2026
- **Feb 22, 2026** — Paper accepted 🎉  
- **Feb 23, 2026** — Paper released on arXiv
-  **Apr 12, 2026** — Code released



## Cite
If you find this repository useful in your research/project, please consider citing the paper:
```bash
@article{cai2026raid,
  title={RAID: Retrieval-Augmented Anomaly Detection},
  author={Cai, Mingxiu and Zhang, Zhe and Wu, Gaochang and Chai, Tianyou and Zhu, Xiatian},
  journal={arXiv preprint arXiv:2602.19611},
  year={2026}
}
```

