# RAID: Retrieval-Augmented Anomaly Detection
### [Project Page](https://github.com/Mingxiu-Cai/RAID) | [Paper]()

The official repository for **RAID: Retrieval-Augmented Anomaly Detection**. Code coming soon!

![Image](https://github.com/Mingxiu-Cai/RAID/blob/main/images/pipeline.png)
We propose RAID, a retrieval-augmented UAD framework designed for noise-resilient anomaly detection and localization. Unlike standard RAG that enriches context or knowledge, we focus on using retrieved normal samples to guide noise suppression in anomaly map generation. RAID retrieves class-, semantic-, and instance-level representations from a hierarchical vector database, forming a coarse-to-fine pipeline. A matching cost volume correlates the input with retrieved exemplars, followed by a guided Mixture-of-Experts (MoE) network that leverages the retrieved samples to adaptively suppress matching noise
and produce fine-grained anomaly maps.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
