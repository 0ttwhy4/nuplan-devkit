# Architecture of Hydra-MDP

![Hydra-MDP architecture](./hydra-mdp.png)

## Perception Network

**You can see Transfusion for implement details**

The perception network builds upon Transfuser, which consists of an image backbone, a LiDAR backbone and perception heads for 3D object detection and BEV segmentation. Multiple transformer layers connect features from stages of both backbones.

### PointCloud

LiDAR points of 4 frames splatted on the BEV plane, encoded by ResNet34

### Image

Concatenated Image(256x1024) -> [ImageBackbone:ResNet34] -> Image tokens

### Modality Fusion

LiDAR Tokens + Image Tokens -> [CrossAttention] -> Env Tokens

### Transfusion

All the perception process above is performed by [Transfusion] module. Transfusion module takes in 

## Trajectory Decoder

Planning Vocabulary (Q) + Env Tokens (K,V) -> [TransformerDecoder] -> 1. [ScoreMLP] -> Imitation Score

-> 2. [MetricMLP] -> Metric Scores

1. Construct Planning Vocabulary

Sample 700K trajectories randomly from the original nuPlan database, each of which consists of 40 timestamps of (x,y,heading). The planning vocabulary $\mathcal{V}_k$ is formed as K-means clustering centers of the 700K trajectories.

2. Vocabulary Embedding

**You can see VADv2 for implement details**

$\mathcal{V}_k$ is embedded as $k$ latent queries with an MLP, sent into layers of transformer encoders and added to the ego statues E:
$$\mathcal{V}_k' = Transformer(Q,K,V=Mlp(\mathcal{V}_k)) + E$$
Then the embedded query interacts with the environmental feature through cross attention
$$\mathcal{V}_k'' = Transformer(Q=\mathcal{V}'_k,K,V=F_{env})$$
The loss function of imitation is
$$\mathcal{L}_{im}=-\sum\limits_{i=1}^k y_i \log(\mathcal{S}_i^{im})$$
where $\mathcal{S}_i^{im}$ is the $i$-th softmax score of $\mathcal{V}''_k$ and $y_i$ is the imitation target produced by L2 distances between log-replays and the vocabulary, which is produced by:
$$y_i = \frac{e^{-(\hat{T}-T_i)^2}}{\sum_{j=1}^k e^{-(\hat{T}-T_j)^2}}$$ 