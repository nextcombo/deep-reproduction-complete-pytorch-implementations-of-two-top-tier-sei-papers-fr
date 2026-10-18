# Deep Reproduction | Complete PyTorch implementations of two top-tier SEI papers: from self-supervised clustering to recurrent neural network compression [with Python code]

Deep Reproduction | Complete PyTorch implementations of two top-tier SEI papers: From Self-Supervised Clustering to Complex Neural Network Compression SCSC (IEEE IoT-J 2023) + CVCNN (IEEE JSAC 2021) complete reproduction and analysis summary.

Specific Emitter Identification (SEI) is a core technology in wireless physical layer security, which achieves precise identification of communication devices through extracting radio frequency fingerprints.This article fully reproduces two SEI deep learning methods published in IEEE top-tier journals: (1) SCSC—an unsupervised SEI method based on signal contrastive self-supervised clustering; (2) CVCNN—an efficient supervised SEI method based on multi-value convolutional neural networks and network compression. The complex modernization is entirely built from scratch, providing a synthetic dataset generator for end-to-end verification, supporting standard public datasets (such as RML2016) and plug-and-play access to custom IQ data.Experimental results show that the SCSC method can improve ACC from 0.125 to 0.64 and NMI to 0.67 in just 40 training rounds when synthesizing an 8-source scene. After triple-S pruning, CVCNN (N=64) achieves a 57.8% parameter compression and a 53.0% reduction in computational load without sacrificing accuracy. Moreover, the model output after pruning is mathematically equivalent to the masked model (max|Δ|≈1.9×10⁻⁶), fully verifying the reliability of the replication scheme and the correctness of the theoretical basis presented in the paper.
Keywords: Specific Radiation Source Identification, Comparison of Self-Supervised Clustering, Multi-Value Convolutional Neural Networks, Network Compression, Knowledge Distillation, PyTorch Reproducibility
I. Introduction With the rapid development of wireless communication technology and the explosive growth of Internet of Things (IoT) devices, wireless network security faces unprecedented challenges. Although traditional cryptographic-based top-level authentication mechanisms are widely used, they often fall short in resource-constrained edge devices and complex electromagnetic environments. Specific Emission Source Identification (SEI) technology, also known as Radio Frequency Fingerprinting (RF Fingerprinting), achieves device identification by analyzing the "unintentional modulation" characteristics introduced by hardware manufacturing processes in the signals transmitted by wireless devices, which possesses unique advantages in physical layer anti-counterfeiting. Therefore, it has important application value in military communication security, IoT device authentication, and spectrum management.
In recent years, deep learning technology has demonstrated strong feature extraction capabilities in the SEI field. However, two core challenges exist in this area: firstly, the scarcity of labeled data - a large amount of unlabeled radio frequency signals in real scenarios is difficult to fully utilize; secondly, the requirement for model deployment real-timeness - SEI tasks usually need to be run on resource-constrained edge devices, which impose strict constraints on the model's parameter size and computational requirements.
In response to the above challenges, Hao and others proposed the SCSC (Signal Contrastive Self-Supervised Clustering) method in a paper published in IEEE Internet of Things Journal (IoT-J) in 2023. This method introduces the contrastive self-supervised clustering framework for the first time into the field of one-dimensional radio frequency signals, achieving radiation source clustering without labeled data.In the paper published in IEEE Journal on Selected Areas in Communications (JSAC) in 2021, Wang et al. proposed a method that combines CVCNN (Complex-Valued CNN) with Triple-S network compression and knowledge distillation to achieve model lightweightness while maintaining high recognition accuracy.
However, due to the original RF datasets of both papers not being publicly available, and some precise details of hyperparameters and network architectures not fully recoverable from PDFs, this has made it difficult for researchers in the community to follow up on their work. This paper faithfully reproduces these two methods, building a complete PyTorch implementation from scratch, and verifies each pipeline end-to-end with a carefully designed synthetic radiation source dataset, while retaining the ability to access real/public data, providing a reliable foundation for subsequent research practices.
2.1 Core idea of Specific Radiation Source Identification (SEI) technology: The core idea of SEI technology is that hardware differences are inevitable in every wireless transmitter during its manufacturing process—including crystal oscillator frequency offset (CFO), I/Q branch gain/phase imbalance, power amplifier (PA) nonlinear distortion, carrier leakage, etc. These "fingerprints" of hardware are subtle but stable, which can be used as the unique identifier of device identity. Traditional SEI methods rely on manual feature extraction (such as instantaneous frequency, power spectral density, bispectrum analysis, etc.), which is laborious and has limited generalization capabilities. Deep learning methods achieve significant performance improvements over traditional methods by automatically extracting discriminative features through end-to-end learning.
2.2 Self-supervised contrastive learning: Contrastive learning has achieved revolutionary breakthroughs in the field of computer vision in recent years, with the core idea that: by applying different data augmentations (such as cropping, rotation, color jittering, etc.) to the same data sample, two different "views" (view) are generated. The model learns to bring two views of the same raw sample closer in the feature space while pushing apart the views of different samples. The success of classic frameworks like SimCLR and MoCo proves the effectiveness of this paradigm. However, applying contrastive learning to one-dimensional RF signals faces two key challenges: how to design suitable data augmentation strategies for signal characteristics, and how to adapt the single-dimensional temporal structure of the signal.
2.3 Complex-Valued Neural Networks (CVNs) directly compute in the complex domain, naturally preserving both the amplitude and phase information of the baseband signal. Deep Complex Networks proposed by Trabelsi et al. at ICLR in 2018 provide a theoretical framework for CVNs, including complex convolution, complex batch normalization (using 2×2 covariance whitening), and CReLU activation functions. CVCNN systematically applies this framework to SEI tasks, serving as a paradigmatic application of CV deep learning methods in RF signal processing.
2.4 Network compression techniques Deep neural network deployment is often limited by computational resources, storage space, and power consumption. Network pruning is one of the main compression methods, which involves removing redundant weights or channels to reduce model size. Triple-S (Sparse Structure Selection) is a structured pruning method that learns to identify the importance of each channel through a triplet mask ({−1, 0, +1}). It achieves high compression ratios while maintaining the output integrity of the pruned model.
III. Method One: SCSC - Signal Contrastive Self-Supervision Clustering
3.1 Overview The core idea of SCSC is to unify the task of radio frequency signal self-supervised contrastive learning with clustering into an end-to-end framework. Unlike traditional two-stage methods that first perform contrastive pretraining and then clustering post-processing, SCSC optimizes both instance-level contrastive loss (Instance-level Contrastive Loss) and cluster-level contrastive loss simultaneously during a single training stage, facilitating mutual promotion and collaborative optimization of feature representation learning and cluster assignment learning.
The entire framework requires only the number of clusters to be clustered (K, i.e., the number of sources), without any prior labeled information.
3.2 Network Architecture
SCSC's network architecture consists of three parts:
(1) 1D-FPFE Backbone Network (1-D Fingerprint Pyramid Feature Extractor)
This is one of the core innovations of SCSC. Unlike the conventional CNN with 3×3 small convolutional kernels, the 1D-FPFE employs large-sized one-dimensional convolutional kernels (15×1, 19×1, 23×1) to capture long-range fingerprint features of signals. Its "pyramid" structure is reflected in: after each convolutional module output, global average pooling (GAP) is performed, and feature descriptors of different scales (with channel numbers of 64, 128, 256) are concatenated into a multiscale feature vector, which is projected to a fixed dimension (e.g., 256-dimension). This multiscale feature fusion mechanism can simultaneously capture the micro-differences and macro-patterns of radio frequency fingerprints.
(2) Instance Head
The instance head is a simple two-layer MLP (linear layer → ReLU → linear layer), which projects the backbone features to a low-dimensional embedding space (e.g., 128-dimension), and obtains an instance feature vector with unit norm through L2 normalization. The instance contrast loss adopts NT-Xent loss (Normalized Temperature-scaled Cross Entropy Loss), which brings two positive sample views generated by different enhancements closer together and pushes the views of different signals further apart.
(3) Cluster Head
The clustering comparison loss also uses NT-Xent, but the operation is from sample space to cluster space: transposing the (N, K) assignment probability matrix to (K, N), making each cluster a basic unit for contrastive learning. To prevent all samples from collapsing into the same cluster, which is a "trivial solution", the cluster loss introduces an entropy regularization term H(Y), encouraging the uniform distribution of cluster assignments.
3.3 Data augmentation strategy: BPS (Bit-Pulse Selection) is a data augmentation method specially designed by SCSC for radio frequency signals. The core idea is to uniformly divide the signal along the time axis into M segments (segment), with each segment determined whether to be enhanced (bit=1 enhance, bit=0 keep) by a random bit, so that the model learns the invariant features of the transmitter under the premise of maintaining "physical consistency".
Four basic enhancement operations include:
SS (Segment Switching): Time reversal of the selected segment to simulate the effect of multipath propagation of signals;
AJ (Amplitude Jitter): Multiplying the selected segment by a random gain σ~U(0.9, 1.1), to simulate channel fading fluctuations;
TS (Timing Skew): Cyclic shifting of the selected segment to simulate sampling clock offset;
RS (Random Noise): Injecting Gaussian white noise SNR~U(1, 15)dB into the selected segment to simulate link noise.
Each forward propagation randomly selects one of the enhancement methods to act on the two views of the signal, thereby constructing positive sample pairs.
3.4 Joint optimization objective The total loss function of SCSC is:
Where:
L_I: Instance-level NT-Xent contrast loss, applied to normalized instance features. For a batch of N samples, there are 2N representations formed by two augmented views, with positive sample pairs being from the same original signal and the remaining 2N−2 as negative samples. The loss is calculated for the cross-entropy of all positive sample pairs.
L_E: Cluster-level contrast loss, consisting of two parts: intra-cluster contrast loss (performing NT-Xent on the cluster space, with positive sample pairs being from the same cluster), and entropy regularization term (−[logK + Σ(p·log p)]), ensuring cluster assignments are uniform and preventing degenerate solutions.
The SCSC paper notes that for RF signals, the temperature coefficient may not be necessary—defaulting to τ=1.0 (cosine similarity) is sufficient, while lower temperatures (e.g., τ=0.5) can accelerate convergence on synthetic data.
3.5 Design of the synthetic radiation source dataset due to the lack of access to the original dataset, this project has designed a synthetic radiation source dataset based on the signal physical model, aimed at end-to-end validation of the effectiveness of all pipelines.
Design principles: All radiation sources emit identical known waveforms (simulating LoRa/WiFi preamble codes), distinguished only through hardware damage parameters. Specifically includes:
Carrier frequency offset (CFO): uniformly distributed between [−0.15, +0.15]
IQ gain imbalance: randomly distributed between [0.85, 1.15]
IQ phase imbalance: randomly distributed between [−0.20, +0.20] radians
PA third-order nonlinear coefficient: randomly distributed between [−0.10, +0.10]
Carrier leakage (DC): complex random, amplitude ≤ 0.05
Each instance injected with random channel phase (U(0, 2π)), slight time jitter (0-7 sample points cyclic shift), and AWGN (15-25 dB SNR), ensuring the model focuses on the emission source fingerprint rather than signal power through instance-level normalization.
4.2 Complex Convolutional Neural Network (CVCNN)
CVCNN operates directly on complex baseband signals, utilizing the real-part-imaginary-part separation storage format: a feature map representation with C channels of complex numbers, length L, is represented as a real tensor of shape (B, 2C, L), where the first C channels are for the real part and the last C channels are for the imaginary part.
Key components include:
1. ComplexConv1d: A convolutional layer that implements complex multiplication -
Using two separate real-number 1D convolutions to implement real and imaginary weight matrices.
2. ComplexBatchNorm1d with Covariance Whitening: This is a key design in complex value networks. Simply applying batch normalization separately for the real and imaginary parts would ignore their correlation. CVCNN employs the 2×2 covariance whitening proposed by Trabelsi et al.: calculating the 2×2 covariance matrix between the real and imaginary parts, which is given by [[V_rr, V_ri], [V_ri, V_ii]], and then using matrix square root inverse to whiten, followed by a learnable complex affine transformation. This ensures that the normalized real and imaginary parts are uncorrelated after normalization.
3. CReLU activation function: Applying ReLU independently to the real and imaginary parts, i.e.:
(4) Complex Max Pooling (ComplexMaxPool1d): This is a unique design of CVCNN - selecting the sample point with the maximum amplitude (|z|² = r² + i²) within each pooling window, and retaining both real and imaginary parts simultaneously instead of separately pooling the real and imaginary parts. Such "true complex operation" ensures phase consistency after pooling.
Network architecture: 9 layers of stacked convolutions (kernel size 3, padding SAME) + max pooling (stride 2) alternating → 2 fully connected layers (the first layer is a complex linear layer + CReLU, the second layer is a real number classification layer for amplitude vectors). The network width is controlled by parameter base_width (i.e., N in the paper, such as 64 or 128), all convolutional layers use the same width.
4.3 Complexity Features: CVCNN vs RVCNN The core advantage of CVCNN lies in its parameter efficiency. The complexity measurement on synthetic datasets (signal length 1024, 10 classes of radiation sources) shows the same pattern as the paper:
Model
Parameter count
MACCs  
CVCNN (N=64)  
235,338  
50,988,288  
CVCNN (N=128)  
862,346  
202,245,376  
RVCNN (N=128)  
431,626  
50,955,520  
Three important rules:
The MACCs of CVCNN(64) approximately equal RVCNN(128), meaning that the complex value network only requires half the width to achieve equivalent computational capacity as a real-valued network.
CVCNN(128) ≈ 4× CVCNN(64) MACCs, consistent with the quadratic growth of complex value operations.
The parameter count of CVCNN(64) is less than RVCNN(128), indicating that the revalued parameters are naturally more compact.
4.4 Triple-S sparse structure selection pruning The core of Triple-S is to introduce a learnable ternary mask parameter vector m for each convolutional layer. Each value corresponds to a complex channel. During the training process, Mask through "Proximal Straight-Through Gradient Descent, PSTGD) "update, Convolutional weights are still optimized by standard SGD/Adam.
Mask update rules:
1  
Layer-by-layer proximal update:
Among them, prox is the L1 soft threshold operator - soft_threshold(x, thr) = sign(x)·max(|x|−thr, 0), which implements L1 sparse regularization.
2  
Constraint projection: m ← clip(m, -1, +1);
3  
Layer-wise scaling: m ← m / maxⱼ|mⱼ|, to prevent all mask values from shrinking in proportion to L1 regularization.
During forward propagation, the mask is quantized into three-valued:
The threshold hyperparameter (default 0.5) is μ. The quantized direct estimation (STE) ensures that the gradient can be passed back through Htanh (where the gradient is 1 when |m|≤1, otherwise 0).
Physical pruning: After training, remove all channels with a mask of 0 to construct SlimCVCNN. The sign (+1 or -1) of the mask for surviving channels is multiplied with the convolutional weights corresponding to the next layer, ensuring that the forward output of SlimCVCNN is mathematically equivalent to the original mask model—an empirical verification shows that the maximum output difference max|Δ| ≈ 1.9×10⁻⁶, which is merely a floating point cumulative error.
4.5 The knowledge distillation compressed model may have a decrease in accuracy under certain conditions (especially low SNR). The knowledge distillation stage adopts the teacher-student framework:
The teacher model is a uncompressed wide CVCNN (e.g., N=128), while the student model is SlimCVCNN. The distillation loss L_T supports four metrics: cross-entropy (CE), KL divergence (KL), mean squared error (MSE), and Jensen-Shannon divergence (JS). The soft label temperature T controls the smoothness of the predicted distribution. The teacher model is frozen throughout the distillation process.
V. Experiments and Analysis 5.1 Experimental Setup Hardware Environment: All experiments can be completed solely on the CPU (training time varies from a few minutes to several hours, supported by CUDA acceleration).
Dataset:
Synthetic radiation source dataset: Default 8 radiation sources, 120 signals per source, length 512, for rapid validation.
Expanded synthetic data: 10 radiation sources, 300 signals per source, length 1024, for compression experiments.
Supports RML2016 and custom.npy data
Evaluation indicators:
Clustering task: ACC (HungarianAlgorithm alignment), NMI (Normalized mutual information), ARI (Adjust the Land Index)  
Classification task: Accuracy
Compression task: Parameter compression ratio R_param, calculation compression ratio R_macc.
Visualization: t-SNE feature dimensionality reduction projection
5.2 SCSC Clustering Experiment Configuration: Synthetic dataset, 8 radiation sources, 120 signals per source, signal length 512, batch size 128, training 40 rounds, instance temperature τ=0.5.
Experimental results:
SCSC started from ACC=0.125 in random clustering and achieved the following results after 40 rounds of training:
Metrics
Initial (random)
After 40 rounds.
ACC  
0.125  
0.640NMI  
—  
0.670ARI  
—  
0.500  
Key observation:
(1) Convergence trend health: The indicators still show an upward trend after 40 rounds, indicating that the model has not yet overfitted, and further training could significantly improve it. The end-to-end joint optimization design of SCSC enhances both clustering quality and feature representation quality mutually.
(2) Effectiveness of BPS enhancement: The combination of four enhancement methods (SS/AJ/TS/RS) provides rich "invariance" signals for the model, allowing it to learn to ignore "disturbance changes" such as channel noise and time jitter while retaining "discriminative differences" reflecting hardware fingerprints.
(3) Influence of temperature parameter: The paper reports that τ=1.0 is sufficient, but in practice, training on a small synthetic dataset with τ=0.5 was faster—a lower τ makes the contrastive loss "harder," enhancing the diagonal advantage of the similarity matrix, thereby accelerating the separation of the feature space.
(4) t-SNE visualization: The feature dimensionality reduction projection clearly shows the radiating source clustering structure, where each radiating source forms separate clusters in the embedding space, corroborating with the clustering metrics.
Experimental configuration for CVCNN classification and compression: Synthetic dataset, 10 radiating sources (compressed experiment reduced to 6 sources), 300 signal samples per source, length 1024, SNR = 30 dB.
Basic classification results (CVCNN trained 60 rounds):
CVCNN and RVCNN baselines achieved perfect or near-perfect classification accuracy (ACC ≈ 1.000), proving the effectiveness of complex value feature extraction in SEI tasks. Under high SNR scenarios, both architectures can reliably extract fingerprint features of emission sources.
Triple-S compression results (μ=0.5, λ_SR=3):
After applying Triple-S pruning to CVCNN (N=64), the surviving channel distribution is: [47, 44, 46, 33, 38, 39, 43, 40, 36] (all original 9 layers were 64 channels), key statistics are as follows:
Metrics
Before pruning
After pruning
Compression rate
Parameters:
1.00%  
42.2%  
↓57.8% MACCs  
1.00%  
47.0%  
↓53.0% Classification Accuracy  
1.000  
1.000  
Non-destructive  
Key insights:  
(1) The beauty of lossless compression: After pruning, the SlimCVCNN not only retains 100% accuracy but also, mathematically—verified rigorously—the difference between its forward output and the masked model is merely max|Δ| ≈ 1.9×10⁻⁶. This means that triple masking is not just a "soft" approximation but an exact equivalent transformation. This result is crucial for practical deployment: it ensures that the precision of your trained model does not suffer due to model simplification during deployment.  
(2) Controllable compression magnitude: By adjusting λ_SR (L1 sparsity strength) and μ (triple threshold), one can flexibly balance compression rate and accuracy. A larger λ_SR encourages stronger sparsity, while μ controls what weights are considered zero—two hyperparameters jointly determine the final channel survival rate.  
(3) Value of knowledge distillation: In high SNR scenarios, compression itself may not lead to precision loss, and the role of KD is not significant. However, in low SNR conditions (such as SNR < 10dB), compression may remove some "weak channels" that are useful for noise robustness, and the ability of KD from the teacher model to transfer soft knowledge to the student model is crucial for restoring low SNR accuracy.  
(4) The role of mask initialization noise: When all channel statistical characteristics are the same (e.g., trivial data), masks may fall into symmetry degradation—all mask values being 1.0 and not distinguishable by gradients. By adding small perturbation noise (default σ=0.3) to break this symmetry, Triple-S can then proceed with structured learning.  
5.4 Reproducibility and limitations of reproducibility confirmation items:  
✅ Complete reproduction of SCSC's end-to-end clustering framework (1D-FPBE + dual contrast heads + BPS enhancement)
✅ Complete re-implementation of CVCNN's multivalue computation framework (multi-convolution/BN whitening/CReLU/magnitude pooling)
✅ Complete replication of Triple-S's PSTGD mask update mechanism
✅ Complexity relationship consistent with the paper (MACC/parameters comparison between CVCNN vs RVCNN)
✅ The output equivalence of structured pruning is mathematically verified.
✅ Knowledge distillation supports four metrics: CE/KL/MSE/JS
Known Limitations:
The RF dataset for the paper is not publicly available, and to reproduce the results, synthetic datasets are used to validate the method's effectiveness rather than its absolute performance figures.
CVCNN table I's precise channel configuration and hyperparameters could not be recovered from the PDF, adopting parameterized design (base_width control).
SCSC's open set rejection (ZSL+ semantic centroid) and CVCNN's precise table I scheduling have not been implemented yet.
VI. Discussion 6.1 Inspirations at the methodology level SCSC and CVCNN have chosen different technical approaches in the SEI task - the former is unsupervised clustering, The latter is supervised classification + compression - but both share a profound insight: the special physical structure of radio frequency signals requires specially designed neural network architectures. The large kernel 1D convolution of SCSC is adapted to the long-range correlation of signals. The multi-domain operation of CVCNN is a respect for the natural complex structure of IQ signals. These "domain knowledge-driven" design choices. Instead of the generalized "take and use" architecture. It is the key to their success.
6.2 Value in Engineering Practice: From an engineering perspective, SCSC and CVCNN address two key requirements for SEI deployment respectively.
SCSC has resolved the "labeling cost" issue - in actual SEI tasks, it is unrealistic to collect a large number of labeled samples for each new radiation source. Unsupervised clustering can directly discover the structure of radiation sources from unlabeled captured signals.
CVCNN+Triple-S+KD addresses the "deployment" issue—even with high-precision models, running them on edge devices with limited energy consumption, latency, and storage remains a challenge. By combining structured pruning and knowledge distillation, the model can compress by more than half while maintaining accuracy, making edge deployment feasible.
The academic significance of "reproduction" (reproduction)—the ability to independently implement methods from the original author's data and code, obtaining comparable results—is an indispensable part of scientific methodologies. The contribution of this project lies in: when the original RF dataset is unavailable, constructing a reasonable alternative dataset through physical modeling to verify whether the core mechanism of the method holds; when there are ambiguities in the technical details of the paper, transforming the method into runnable code through reasonable inference and parameterization design. All this provides a complete path for peer researchers to "read the paper → understand → try → verify → improve."
Conclusion and Future Prospects This paper faithfully reproduces the core methods of two top SEI papers and verifies them end-to-end through synthetic datasets. The main contributions include:
1  
Complete reproduction of the SCSC contrastive self-supervised clustering framework—1D-FPFE backbone, BPS signal enhancement, dual head joint optimization, achieving clustering performance of ACC=0.64 and NMI=0.67 on 8 radiation source synthetic data, validating the feasibility of unsupervised methods on SEI tasks.
2  
Complete reproduction of the supervised classification and compression pipeline of CVCNN+Triple-S+KD—the revalued CNN directly processes IQ complex baseband signals, Triple-S structural pruning achieves 57.8% parameter compression and 53.0% computational reduction without sacrificing accuracy, and after pruning, the model is mathematically equivalent to the original model, validating the technology path for lightweight SEI deployment.
3
A physical model-based synthetic radiation source dataset generator has been constructed, supporting the simulation of real hardware damages such as CFO, IQ imbalance, PA nonlinearity, and carrier leakage. This provides a publicly reproducible verification benchmark for the community.

The complete PyTorch implementation code and detailed documentation have been provided, supporting the flexible access to synthetic data, RML2016 public datasets, and custom IQ data.

Future work directions include: (1) implementing the open set identification module (ZSL + Semantic Centroid) of SCSC; (2) restoring the precise table I scheduling of CVCNN; (3) conducting large-scale verification on more real RF datasets (such as USRP collected data); (4) exploring the combination of SCSC and CVCNN - that is, utilizing SCSC's unsupervised feature representation to initialize the supervised classifier of CVCNN.

References [1] Hao, X., Feng, Z., Liu, R., Yang, S., Jiao, L., & Luo, R. (2023). Contrastive Self-Supervised Clustering for Specific Emitter Identification. IEEE Internet of Things Journal, 10(23), 20803-20817.
[2] Wang, Y., Gui, G., Gacanin, H., Ohtsuki, T., Dobre, O. A., & Poor, H. V. (2021). An Efficient Specific Emitter Identification Method Based on Complex-Valued Neural Networks and Network Compression. IEEE Journal on Selected Areas in Communications, 39(8), 2305-2317.  
[3] Trabelsi, C. et al. (2018). Deep Complex Networks. International Conference on Learning Representations (ICLR).  
[4] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. International Conference on Machine Learning (ICML).  

## Images

![Deep Reproduction | Complete PyTorch implementations of two top-tier SEI papers: from self-supervised clustering to complex value neural network compression [with Python code]](cover.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
