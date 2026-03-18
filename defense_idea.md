Here is the defense idea converted into a Markdown file.

# Mitigating Self-Supervised Learning Backdoor with a Single Matrix

## Abstract

Stealthy SSL backdoor attacks, such as DRUPE and INACTIVE, evade detection by mimicking the statistical distribution of clean features. However, we identify a critical geometric vulnerability in these attacks: while aligning with clean distributions in high-rank subspaces, these attacks exhibit a strong and consistent signal in low-rank subspaces. We propose a resource-efficient defense based on this finding and statistical whitening transformation. This transformation detects the poisoned samples by normalizing natural semantic variance and amplifying the backdoor signal to be distinguishable. Experiments demonstrate that our approach detects state-of-the-art attacks with near-perfect accuracy (ROC-AUC > 0.99).

## 1. Introduction

Self-supervised Learning (SSL) has fundamentally transformed the landscape of artificial intelligence. By mitigating the reliance on expensive labeled datasets, SSL enables models to extract robust representations directly from raw, unlabeled data. This paradigm has facilitated the rise of foundational models capable of adapting to diverse downstream tasks with minimal supervision. For instance, OpenAI's CLIP demonstrates that pre-training on 400 million uncurated text-image pairs yields strong zero-shot performance across various classification benchmarks. Similarly, contrastive learning frameworks like SimCLR have proven that representations learned from unlabeled data can match or even exceed the performance of fully supervised baselines, establishing SSL as a standard for visual feature extraction.

However, this paradigm introduces a critical security vulnerability. SSL models are typically pre-trained on vast corpora of data scraped "in the wild" without rigorous curation. This reliance on untrusted sources creates ideal conditions for backdoor attacks, where an adversary injects specific triggers (e.g., pixel patterns or semantic patches) into the training corpus. During pre-training, the model learns to associate these triggers with a target behavior while maintaining normal functionality on clean inputs, rendering the attack stealthy.

This security gap translates into a severe practical risk due to the "train once, deploy everywhere" nature of the modern AI supply chain. Because pre-training requires substantial computational resources, practitioners often rely on third-party weights from open repositories. This creates a single point of failure: a backdoor injected into a widely used encoder acts as a dormant virus that persists through downstream fine-tuning. Consequently, security-critical applications—such as biometric access control—can be compromised not by attacking the application directly, but by poisoning the underlying foundation model to bypass authentication via a stealthy trigger.

Recent literature confirms that SSL encoders are highly susceptible to such supply-chain poisoning. Attacks such as BadEncoder demonstrate that backdoors injected into a pre-trained image encoder are effectively inherited by downstream classifiers. More sophisticated attacks, such as invisible-trigger attacks and embedding-space attacks, claim to be indistinguishable from clean models by human perception and standard verification metrics, posing a significant challenge for mitigation. Furthermore, defenses designed for supervised learning are often inapplicable to SSL due to fundamental differences in how the two paradigms organize feature spaces. While some SSL-specific defenses exist, they often incur high computational costs such as training auxiliary decoders or fail against state-of-the-art (SOTA) stealthy attacks that optimize for invisibility. This necessitates a defense mechanism that is both computationally efficient and grounded in a robust theoretical foundation to generalize against unseen threats.

By analyzing the embeddings properties of recent SOTA backdoor attacks on SSL, specifically DRUPE and INACTIVE, we identify a critical, inherent trade-off in attack design. To ensure stealthiness, these attacks must avoid disrupting the dominant semantic features of the data, which correspond to the high-variance principal components of the clean distribution. However, to ensure effectiveness, the attack must embed a rigid, consistent pattern that the model can reliably recognize. Our findings reveal that attackers resolve this trade-off by hiding trigger artifacts in the low-variance dimensions of the clean data's embedding space. In these "silent" subspaces, the trigger signal creates a strong statistical anomaly that is invisible to defenses focusing on the primary semantic (high-variance) axes.

Existing defenses typically fail to exploit this characteristic; they operate on the assumption that poisoned samples form distinct clusters or distance-based outliers in the primary embedding space. Advanced attacks invalidate this assumption by aligning triggers with benign distributions in high-variance dimensions, effectively masking their presence from standard detection methods.

To address this, we propose a novel, lightweight defense mechanism based on ZCA Whitening. Our approach leverages the embeddings insight described above: while clean semantic signals dominate high-variance dimensions, trigger signals are concentrated in low-variance dimensions. We prove that a whitening transformation derived from a small, trusted reference dataset can be applied to normalize the variance of all dimensions to unity. This transformation effectively "whites out" the dominant clean signals and mathematically forces the trigger signals hidden in low-variance subspaces to be disproportionately amplified. Consequently, previously stealthy backdoors are converted into statistical outliers. Our method requires no model retraining, no knowledge of the specific attack pattern, and introduces negligible computational overhead.

We validate our approach against multiple SOTA SSL backdoor attacks, including BadEncoder, DRUPE, INACTIVE, and BadCLIP, across benchmarks such as CIFAR-10 and ImageNet. Empirical results demonstrate that our whitening-based defense achieves high detection performance (often exceeding 99% AUC), effectively neutralizing the backdoor threat while preserving model utility.

In summary, this paper makes the following contributions:

* 
**Embedding Analysis of SSL Backdoors:** We provide a detailed analysis of the embedding space of poisoned SSL encoders, revealing that effective backdoor triggers inevitably concentrate in the low-variance subspaces of the data distribution to maintain stealth.


* 
**Theoretical Foundation for Whitening Defense:** We formally derive the "Whitening Amplification" theorem, proving that a whitening transformation  constructed from clean data covariance naturally amplifies the  norm of trigger artifacts relative to clean samples ().


* 
**A Lightweight Backdoor Defense:** We propose a practical, inference-time detection algorithm that distinguishes clean and poisoned samples via projection onto the amplified backdoor axes. Extensive experiments show our method outperforms existing baselines in detecting invisible attacks with minimal computational cost.



## 3. Threat Model and Assumptions

### 3.1 Attacker

**Attack Objective:** The attacker aims at achieving backdoor effect by manipulating the poisoned encoder's image embeddings with triggers, such that when a downstream classifier is built on the embeddings from this encoder, it will classify the triggered embedding as the target class chosen by the attacker. Furthermore, the attack must be stealthy when the trigger is not presented, that is, the encoder should produce a natural embedding from a normal image without trigger, and thus the downstream classifier built on these benign embeddings should have similar performance as building from a clean encoder's embeddings. Note that an ideal attack should still be able to manipulate the downstream prediction even when the classifier is trained on the embeddings from a clean dataset via the poisoned encoder.

**Attacker Knowledge and Capabilities:** Following prior works, we consider the attacker to have full access to the clean pre-trained encoder and a shadow dataset, which will be used as the attacker's training dataset for injecting malicious samples. The attacker also has enough computing resources to perform finetuning the pre-trained encoder. On the other hand, the attacker does not have any access to the downstream dataset, training process or model weights/architecture of the classifier, except for very few images of the target class, mentioned as reference images. After successfully injecting the trigger, the attacker might release the poisoned model's weights as a frozen pre-trained model to the defender to perform malicious actions.

### 3.2 Defender

**Defense Objective:** The defender's main goal is to determine if the acquired pre-trained model is poisoned or not, and if it is, nullify the trigger's effect during downstream classification by detecting the poisoned samples during inference to prevent the attacker from manipulating the prediction results with triggers. The detection and mitigation of backdoor should be performed under limited resources, while ensuring generalizability across a range of attacks and future unseen attacks.

**Defender Knowledge and Capabilities:** The defender has no knowledge about the attacker's methods, trigger pattern or target class, while having full control to the downstream dataset, downstream classifier training process and model weights/architecture. Contrary to the attacker, the defender does not have enough resources for finetuning the encoder or training the encoder from scratch, thus their defense should be as less resource-intensive as possible.

**Assumptions about the Defender:** To perform backdoor defense, the defender assumes to have access to a small, trusted clean dataset that is often a fraction of the examining encoder's pretrain dataset. This assumption is a common practice condition for studying SSL backdoor defense. In our experiments, we found that taking 1% of pretraining dataset is reasonable and empirically feasible.

## 4. Motivation

Our approach is built upon a finding that is essentially different from prior defenses assumption, which provides critical insights about the current attacks' success factor. It is commonly believed among current defenses that the effectiveness of SSL backdoor attack depends on the proximity of the backdoored embeddings to the target class embeddings and benign class embeddings, forming the core axiom for defenders to distinguish and mitigate backdoor attacks in SSL. This effectiveness is acquired by the attacker by forcing the encoder to shift its mapping so that the backdoored embeddings move closer towards the target embeddings, while maintaining stealthiness by staying close enough to the original cluster. However, we discovered that, beside proximity, there are other strong embedding artifacts sneaking inside the embedding that signal the downstream classifier to recognize and learn the backdoor shortcut from them without relying on the target class related information.

### 4.1 Findings

**Label Flipping Experiment:** To study the success factor of SSL backdoor attack, we conducted an experiment to examine the behavior of downstream classifier under the presence of triggered sample. The attack used during dissecting is DRUPE, a state-of-the-art attack with high effectiveness and stealth. The experiment is designed as below:

1. 
**Dataset acquiring:** we construct a training dataset including benign samples of all available classes, and a number of poison samples with trigger stamped.


2. 
**Target label randomization:** we randomize the labels of these poison samples to a random class, while retaining other benign samples' labels unchanged. Note that in this randomization, all poisoned samples always share the same label.


3. 
**Extract embeddings:** We then feed these images to the backdoored encoder to extract their embeddings. At this point, we have constructed a dataset which looks similar to a downstream classifier training dataset, except for the addition of malicious samples, which have their target labels randomized.


4. 
**Classifier training:** We train multiple simple classifiers, respectively Decision Tree, Support Vector Machines (SVM), Logistics Regression and Multilayer Perceptron (MLP) classifier, on the extracted embedding dataset and observe the result.



**Result:** The table below shows the classification result of downstream classifier when trained with the experimented dataset. If the attacker's chosen target class has a strong impact on the prediction of malicious embeddings, i.e. the poisoned embeddings are trained to specifically look similar to the target reference embeddings so that it is classified as target class during downstream prediction, randomizing the poisoned embeddings labels will disrupt their classification. However, as shown in Table 1, the experiment delivered an opposite result: with any newly assigned label, the classifier's accuracy in predicting poisoned embeddings consistently stays at roughly 99%. This suggests that there must be some label-agnostic artifacts in the embedding that signal the classifier to form the classification shortcut so that the classifier, despite being simple, even linear in the case of Linear Regression, can recognize and predict the poisoned embeddings effortlessly.

**Table 1: Performance of simple downstream models with target label randomized for poisoned samples** 

|  | Poison Accuracy | Clean Accuracy |
| --- | --- | --- |
| **Decision Tree** | 98.87% | 88.51% |
| **SVM** | 98.92% | 89.51% |
| **MLP** | 99.17% | 88.13% |
| **Linear Regression** | 98.84% | 89.24% |

**Embedding Artifact Findings:** To further verify and uncover the suggested artifacts, we employed Principal Component Analysis (PCA) to dissect the embedding of both benign and malicious images. PCA aims at analyzing complex, high-dimensional data by finding a new set of orthogonal axes, called principal components, that capture the maximum variance in the data. The higher the captured variance, the more essential the axes is in representing the information of data.

Analysis of the principal components fitted on the basis of DRUPE's poisoned embeddings demonstrates the top-5 axes with the largest variance captured. The malicious samples distributed plainly on 2 dominant axes, accounted for over 95% captured variance, leaving the rest with approximately no information. On the other hand, the principal components derived from clean embeddings show a set of axes with more evenly distributed information.

This discrepancy in the importance of the principal components between the malicious and clean embedding space further escalates the aforementioned observation: while seems blend-in and stealthy from the perspective of clean principal components, the poisoned embeddings express very strong and consistent signals in the embedding space. This finding suggests an invaluable insights about the pivotal success factor of SSL backdoor attack, while offer a novel axiom of safeguarding pre-trained encoders from backdoor attack.

However, directly applying the PCA discrepancy in detecting backdoor samples is not straightforward:

* Since the defender does not have any knowledge about the poisoned samples or dataset, finding the principal components of poisoned samples is impossible.


* In a loosened setting where the defender can access to a mixed set of clean and poisoned samples in advance, it is still non-trivial to visually distinguish the two distributions, not to mention a solid metrics to quantify the difference.



This situation necessitates a systematic approach backed by a solid theoretical foundation to effectively leverage the aforementioned pivotal finding.

## 4.2 Theoretical Foundation

**Idea:** Derived from the insight about embedding artifact, we can intuitively think that during the learning of SSL embeddings, the encoder has learnt to embed the "signals" about vision concepts and properties in the embedding space. Since the model is trained with a massive number of images and classes with a plethora of information to embed, the signals guide the embedding cloud to form a dense structure spreading every directions, explaining the evenly distributed variance of principal components. In contrast, the trigger stamped to the poisoned images during the attack inject a new strong, consistent signal to the embedding space. Due to the attack being finetuned on a much smaller amount of images and trigger pattern, this new signal is fundamentally different from the normal signal and takes place in a smaller number of particular axes, thus lead to the dominant principal components.

Stem from this intuition, we can think of the malicious embedding as the combination of both clean normal signal and the trigger signal, which differentiate it from other benign samples. The core idea for detecting poisoned samples here is that if we can find a transformation that "white out" the clean signals, i.e. transform clean information to white noise, the malicious samples will expose their trigger signals, since the clean signals have already been cleared. To this end, we formalize the hypothesis and provide a theoretical proof based on the proposed idea to form a foundation for designing the defense mechanism.

**Presumption:** We present a presumption reasoning from the ideal attack's conditions to clarify the setting and set the stage for building up the theoretical foundation. As mentioned in Section III, the attacker's objective is to develop an attack that is both effective and stealthy. Originate from the intuition idea and based on the attack conditions, we can reasonably consider two components of a malicious embeddings as  and , simultaneously satisfy the attack requirements as follow:

* 
** Effectiveness:** This component signals the downstream classifier to recognize and misbehave when the trigger is presented. Since the trigger signal is derived from a small set of images and patterns, this component is often low-variation since the trigger signal is derived from a limited number of reference images and consistent trigger pattern.


* 
** Stealthiness:** This component mimic the characteristics of clean samples to blend the malicious samples to benign distribution. This component achieves stealthiness by spanning the high variance direction of clean samples. These directions represent the "meaningful" features of clean embeddings, thus mimicking clean signal along these direction fulfills the stealthy condition.



Note that for the backdoored encoder to behave normally without the trigger,  and  should statistically independent (or weakly correlated), as the trigger is designed to not disrupt clean semantics significantly. Since the clean component  has already spanned along high-rank dimensions with higher variation in order to mimic the benign features, the trigger component  should sneak in lower-rank dimension with lower variation for stealthy signal.

From there, we can approximate the covariance of poisoned embeddings by the additive relation of clean component and trigger component's covariance:

E_{poisoned} \approx E_{clean} + E_{trigger}
$$ 

From the basis of the aforementioned poisoned embeddings approximation, we state the hypothesis that:

**Hypothesis:** There exists a matrix  that can distinguish the poisoned embedding  and benign embedding  by transforming them such that:

Cov(y) = E[yy^{T}] = E[W x_{b} x_{b}^{T} W^{T}] = W \Sigma W^{T}
$$ 

To satisfy , we choose the whitening matrix . Since calculating  directly is non-trivial, we employ eigendecomposition:

W = \Lambda^{-1/2} U^{T}
$$ 

Applying this  to benign embeddings normalizes their variance in all principal directions to 1, effectively "whites out" the natural semantic clusters.

Now, consider the poisoned embedding . Based on our additive presumption:

||W x_{t}||^{2} = x_{t}^{T} W^{T} W x_{t} = x_{t}^{T} \Sigma^{-1} x_{t}
$$ 

Substituting the eigendecomposition :

||W x_{p}|| \approx ||W x_{c} + W x_{t}|| \gg ||W x_{c}|| \approx ||W x_{b}||
$$ 

This proves that the whitening transformation  disproportionately amplifies the trigger artifact, making it distinguishable from benign embeddings.

## 5. Experiments

We validate our approach against multiple SOTA SSL backdoor attacks, including BadEncoder, DRUPE, INACTIVE, and BadCLIP, across benchmarks such as CIFAR-10 and ImageNet. Empirical results demonstrate that our whitening-based defense achieves high detection performance (often exceeding 99% AUC), effectively neutralizing the backdoor threat while preserving model utility.

**Table 2: Upstream Detection Performance with ImageNet dataset** 

| Defense | BadEncoder |  |  | CTRL |  |  | DRUPE |  |  | INACTIVE |  |  | BadCLIP |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | TPR | FPR | AUC | TPR | FPR | AUC | TPR | FPR | AUC | TPR | FPR | AUC | TPR | FPR | AUC |
| **ASSET** | 84.9 | 4.0 | 0.978 | 89.6 | 30.2 | 0.799 | 94.7 | 27.6 | 0.858 | 0.0 | - | - | 99.8 | 49.4 | 0.773 |
| **DEDE** | 93.1 | 6.9 | 0.981 | 87.2 | 12.8 | 0.912 | 97.6 | 2.4 | 0.997 | 15.3 | 85.8 | - | 85.0 | 14.9 | 0.925 |
| **Ours** | - | - | - | - | - | - | **100.0** | **0.0** | **1.0** | **100.0** | **0.0** | **1.0** | **99.0** | **1.0** | - |



**Table 3: Upstream Detection Performance with CIFAR10 dataset** 

| Defense | BadEncoder |  |  | CTRL |  |  | DRUPE |  |  | INACTIVE |  |  | BadCLIP |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | TPR | FPR | AUC | TPR | FPR | AUC | TPR | FPR | AUC | TPR | FPR | AUC | TPR | FPR | AUC |
| **ASSET** | 81.3 | 9.1 | 0.951 | 86.6 | 32.5 | 0.692 | 92.1 | 28.5 | 0.852 | 0.0 | - | - | 88.7 | 46.4 | 0.693 |
| **DEDE** | 92.1 | 8.2 | 0.961 | 88.3 | 10.8 | 0.903 | 92.6 | 5.4 | 0.927 | 10.1 | 88.0 | - | 85.1 | 13.9 | 0.951 |
| **Ours** | - | - | - | - | - | - | **100.0** | **0.0** | **1.0** | **100.0** | **0.0** | **1.0** | **99.0** | **1.0** | - |
