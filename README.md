# RAID: Retrieval-Augmented Anomaly Detection
### [Project Page](https://github.com/Mingxiu-Cai/RAID) | [Paper]()

The official repository for **RAID: Retrieval-Augmented Anomaly Detection**. Code coming soon!

![Image](https://github.com/Mingxiu-Cai/RAID/blob/main/images/pipeline.png)
We propose RAID, a retrieval-augmented UAD framework designed for noise-resilient anomaly detection and localization. Unlike standard RAG that enriches context or knowledge, we focus on using retrieved normal samples to guide noise suppression in anomaly map generation. RAID retrieves class-, semantic-, and instance-level representations from a hierarchical vector database, forming a coarse-to-fine pipeline. A matching cost volume correlates the input with retrieved exemplars, followed by a guided Mixture-of-Experts (MoE) network that leverages the retrieved samples to adaptively suppress matching noise
and produce fine-grained anomaly maps.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
