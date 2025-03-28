#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices
#import "@preview/tablem:0.2.0": tablem, three-line-table
#import "@preview/showybox:2.0.4": showybox
#import "@preview/subpar:0.2.2"

#set par(justify: true)

#let toprule = table.hline(stroke: 0.08em)
#let botrule = toprule
#let midrule = table.hline(stroke: 0.05em)

#show: arkheion.with(
  title: "Bottled Brilliance: Gated Mixture of Experts for 
  Biomedical Explainability",
  authors: (
    (name: "Sidharrth Nagappan", email: "sn666@cam.ac.uk", affiliation: "University of Cambridge", orcid: "0000-0002-2928-2641"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: [
    The source code is made available at https://github.com/sidharrth2002/biomedical-moe.
  ],
  keywords: ("Biomedical Explainability", "CLIP", "BioMedCLIP", "Mixture of Experts"),
)
#set cite(style: "springer-basic")
#show link: underline

= Introduction

The adoption of deep learning in highly sensitive domains like computational medicine have led to increased calls for robust explainability mechanisms, that medical practitioners can use to trace the reasoning behind specific decisions #cite(<blackbox>). While post-hoc interpretability methods are rampant in the literature, the detachment from the internal workings of the model can result in incomplete explanations #cite(<Rudin2019>), and the "completeness" paradigm is a crucial part of building trust in these automated systems. Concept Bottleneck Models (CBM) organically incentivise models to route decisions through an interpretable concept layer, where each neuron in the bottleneck corresponds to a human-understandable concept. However, annotating concepts can be costly, leading to Language in a Bottle (LaBO) introducing an end-to-end pipeline that leverages Large Language Models (LLMs) to build and enrich concept bottlenecks, before using Vision Language Models (VLMs) such as CLIP to align images and textual concepts #cite(<yang2023language>). Although effective against a range of datasets, performance on the one biomedical dataset they used is among the lowest, having been outperformed by a simple linear probe. 

This raises the question of whether domain knowledge can be implicitly plugged into these models, and whether it can enhance their ability to form robust representations of nuanced datasets—particularly those that rely on subtle morphological cues beyond standard visual descriptors. Following this this line of reasoning, we explore whether the need for domain expertise can be addressed through a mixture of these experts, with each building their own bottlenecks and harmonising representations. Specifically, we question whether combining generalist and specialist models can yield concept bottlenecks that are both performant in end-to-end classification and capable of offering fine-grained, semantically grounded interpretability. 

A learned gating module adaptively combines individual experts, namely the _generalist_ CLIP and the _specialist_ BioMedCLIP. Two representative biomedical datasets for dermatoscopy and radiology are chosen to investigate two fundamental research questions:

1. Does domain-specific expertise improve both the interpretability and classification performance across both biomedical domains?
2. Can this mixture-of-experts outperform single-expert baselines, in both fully supervised and few-shot scenarios?

The findings of this work suggest that MoE-based combinations can produce remarkable boosts in performance during full supervision, while building nuanced, independent bottlenecks that select expert-specific features; this is especially profound in the radiology dataset ($approx 20% arrow.t$ improvement from the generalist baseline). This work also makes methodological improvements to the LaBO pipeline via more structured prompt engineering and straightforward concept extraction.

While evaluating such architectures in few-shot settings introduces complexity, especially given the limited data available to train a robust gating mechanism, it remains a valuable diagnostic tool to see if a model can leverage prior relationships stored in its bottlenecks to act under constrained supervision. In this setting, the specialist expert consistently outperforms its counterparts, with the MoE having unrepresentative gating policies. Though regularisation techniques encourage more balanced expert usage in few-shot settings, further work is needed to enable stable few-shot deployment, such as by transferring gating priors from adjacent biomedical tasks in ways that do not induce leakage.

// Although regularisation techniques encourage more balanced expert contributions during low-shot training, more work is required to transfer learned representations to the gates to scale this method for few-shot deployments, through methods like transfer learning from similar biomedical tasks (that do not induce unwanted data leakage).

// In ultra-low-shot settings, the specialist expert produces the best results, while MoE's lacks sufficient data to learn efficient routing decisions, often yielding unstable results that can sometimes surpass the individual experts. 

// extend this intuition about requiring expertise to mixing experts, and whether combining generalist and specialist experts in the pipeline can work together to build bottlenecks that are not only performant end-to-end, but also granularly interpretable. 

// The adoption of deep learning in computational medicine, particularly in medical image classification, is often hampered by the notion of transparency. Black-box models provide predictions without clarifying the semantic features that drive these decisions, making them difficult to trust in critical environments. Concept Bottleneck Models (CBMs) emerged as an interpretable alternative, that incentivised models to route their decisions through a set of human-understandable concepts. However, these conventional CBMs relied on human annotations to mark those concepts in the first place. Language in a Bottle (LaBO) addressed these challenges by automating concept discovery using Large Language Models (LLMs). These concepts were then aligned to images using pre-trained vision-language models such as CLIP, allowing the formation of a concept bottleneck layer. 

// However, the end-to-end architecture depends on the richness of CLIP's neural representations, whose wide training-base lacks domain-specific grounding. While general visual representations (such as colour and shape) are learnt, they may overlook subtle morphological cues that are essential for deeply nuanced decisions. In contrast, domain-specific models such as BioMedCLIP, trained on scientific literature and imagery, possess specialist knowledge but may lack the broader visual diversity of CLIP. 

// In this mini-project, an extension of the LaBO framework that uses both CLIP and BioMedCLIP as complementary experts, is explored. Specifically, this is framed as a mixture-of-experts (MoE) problem, where CLIP is the generalist expert and BioMedCLIP is the specialist expert, and a learned gating network determines the relative contribution of each for every input image. The motivation is that different skin lesion may benefit from generalist knowledge (e.g. shape, colour patterns) or specialist biomedical cues (e.g. vascular structure, lesion-specific terminology) to varying degrees. A dynamic gating mechanism allows the model to adapatively leverage either or both experts on a per-instance basis, improving flexibility, accuracy, and interpretability. 

// 1. Does domain expertise improve interpretability and classification performance in biomedical image analysis?
// 2. Can a mixture-of-experts fuse information from these disparate concept space, and does this learned combination outperform their uni-expert counterparts?

// We do this across two diverse biomedical datasets, namely in dermatoscopy and radiology. 

//  often outperformed by their more specialised counterparts in highly specialised biomedical tasks, such as computational X-Ray and skin lesion analysis. 

// Transparency is a critical factor, that hampers the adoption of computational models in critical domains. Post-hoc explainability attempts to probe the inner mechanisms of these deep neural networks after they are built, often treating the models themselves as black boxes. Concept Bottleneck Models are therefore used to 

// Advances in vision-language models have revolutionised how machines understand and reason in multimodal settings. Models like CLIP (Contrastive Language-Image Pretraining) have demonstrated remarkable zero-shot and few-shot performance across a range of tasks,

= Related Work

Concept Bottleneck Models (CBMs) improve interpretability by incentivising models to predict human-understandable concepts as an intermediate step before the final prediction #cite(<pmlr-v119-koh20a>) In medical imaging tasks like diagnosing arthritis from an X-Ray, a CBM would first predict clinical concepts (e.g. presence of spurs) and then use those concepts to compute severity. Medical practitioners can inspect and intervene on the model's concept predictions. However, traditional CBMs require training labels for each concept and often lag in accuracy compared to their black-ox counterparts. "Label-free" CBMs transform any network into a CBM without per-concept annotations using rudimentary LLMs #cite(<oikarinenlabel>) for concept discovery. Language In a Bottle (LaBO) extended this paradigm with submodular optimisation to filter relevant and discriminative concepts in the same way a human expert would #cite(<yang2023language>). 

An orthogonal direction leverages vision-language pre-training to tackle limited labels. CLIP is a foundation model that learns joint image-text representations, and have been proven to transfer to new tasks with little or no task-specific data. In the biomedical domain, variants of the CLIP architecture such as BioMedCLIP were proposed, having been trained on vast amounts of scientific text #cite(<biomedclip>).

The Mixture-of-Experts architecture is a long-standing proposition in deep learning, that dynamically combines the strenghts of multiple specialised models using a divide-and-conquer approach #cite(<yang2025mixtureexpertsintrinsicallyinterpretable>) #cite(<eigen2014learningfactoredrepresentationsdeep>). Recent work has applied MoEs to fuse generalist and specialist knowledge, which is particularly relevant in biomedical imaging where a model, much like a doctor, would require both broad and fine-grained expertise. Med-MoE introduced a mixture-of-experts design for medical VL tasks using multiple domain-specific experts alongside a global meta-expert, replicating how different medical specialties unite to form robust diagnoses; it attained state-of-the-art performance by activating only a few relevant experts instead of the entire model #cite(<jiang-etal-2024-med>). Furthermore, because gating decisions reveal which experts were consulted and how much importance was given to their analysis, a clinician can trace deeper intuitions. An Interpretable MoE (IME) uses linear models as experts, with each prediction being accompanied by an exact explanation of which linear expert was used and how it arrived at the outcome #cite(<Ismail2022InterpretableMO>). Impressively, this IME approach maintains accuracy comparable to black-box networks, showing that MoE architectures can incorporate interpretability without sacrificing predictive capacity. 

A tangentially relevant direction uses a hybrid neuro-symbolic design, routing samples down a tree of interpretable experts to explain a black box #cite(<pmlr-v202-ghosh23c>). While most prior efforts apply mixture-of-experts to fully supervised, end-to-end deep networks, we explore its extension to concept bottleneck models—specifically probing whether we can align class-concept association matrices rather than purely combining neural embeddings, and whether this remains effective under few-shot constraints.

// A tangentially relevant work proposes a mixture of interpretable experts, via a hybrid neuro-symbolic model that routes a sample subset down a tree to explain a blackbox #cite(<pmlr-v202-ghosh23c>). Unlike past work that studies and argues for the impact of mixing experts in (i) fully-supervised, (ii) end-to-end deep neural networks, we extend it to concept bottleneck models and question if we can apply the same first principles to align association matrices, instead of simply doing neural combinations, and if these methods have the capacity to be performant in few-shot settings. 

= Method

// == Problem Formalisation

// The objective of the project is to develop an interpretable skin lesion classification model on the HAM10000 dataset, where interpretability is provided through a concept bottleneck layer, in few-shot and fully-supervised settings.

== Biomedical Data

We select (i) HAM10000 (dermatoscopy) and (ii) COVID-QU-Ex (X-Rays) as two representative datasets in the biomedical domain. 

HAM10000 is a collection of 10,015 dermatoscopic images representing seven variations of skin lesions #footnote[melanoma, basal cell carcinoma, and benign keratosis-like lesions] that are compiled from various populations @Tschandl_2018, and is commonly used as a benchmark dataset for medical vision encoders. We use the same training, validation and testing splits as the dataset providers.

// A function to represent a virtual image
#let vimg(body) = {
    rect(width: 10mm, height: 5mm)[
        #text(body)
    ]
}

#figure(
  image("image_samples_ham.png", width: 90%),
  caption: [Sample Images from the HAM10000 dataset]
) <samples>

The COVID-19 Radiography Database comprises 33,920 posterior–anterior chest X-ray images, covering COVID-19, viral/bacterial pneumonia, and normal cases #cite(<RAHMAN2021104319>)#cite(<9144185>). It integrates multiple datasets, including COVID-19 cases from Qatar, Italy, Spain, and Bangladesh, alongside pre-pandemic pneumonia datasets from the USA. Training, validation and test splits are  While we initially considered the NIH ChestX-ray14 dataset #cite(<nih-xray>), its multi-label nature required sigmoid-activated association matrices within our concept bottleneck setup — leading to gradient explosion during training, making it unsuitable for our architecture. 

#figure(
  image("images/covid-qu-samples.png", width: 80%),
  caption: [Sample Images from the COVID-QU-Ex dataset]
) <samples>

// The Covid-19 Radiography database contains 33,920 posterior-anterior chest X-ray images spanning Covid-19, normal and pneumonia-infected samples; it is a combination of COVID-19 and past pneumonia datasets, collected across Qatar, Italy, the USA and Bangladesh. We initially tried employing NIH X-Ray, but the architecture did not scale for multi-label classification that then required sigmoid activations on the association matrices, exploding the loss.  

// #figure(
//     grid(
//         columns: 2,     // 2 means 2 auto-sized columns
//         gutter: 2mm,    // space between columns
//         vimg("1"),
//         vimg("2"),
//     ),
//     caption: "some caption"
// )

== A Representative Bottleneck

LaBO employed sentence parsing using a T5 to extract semantic concepts from LLM-generated sentences #cite(<T5>). We conjecture that this approach is suboptimal, leads to information loss and the quality of the final model is dependent on the accuracy of the trained parsing model. Instead, we propose enforcing JSON structure via Pydantic in prompts we send to our LLM suite (LLAMA, DeepSeek, Meditron and OpenAI's 4o), directly extracting phrasal concepts without intermediate parsing @llama @deepseek. 

Concepts generated for the generalist are augmented with the phrase: "You can be a bit technical." #footnote[After several rounds of prompt engineering, this produced the best results.] Our enhanced prompt engineering outperforms the manual parsing algorithm.

#showybox(
  title: "Technical Prompt Generation",
  frame: (
    border-color: blue,
    title-color: blue.lighten(30%),
    body-color: blue.lighten(95%),
    footer-color: blue.lighten(80%)
  ),
  footer: ""
)[
  Background: Extract the concepts from the class to be used for dermatoscopic images. _You can be slightly technical when generating the concepts._

  Prompt: Describe the {_feature_} of the {_disease_} disease in HAM10000 that can be used as visual concepts for Skin Cancer classification.
]


== Multi-Expert Submodular Optimisation

Submodular optimisation is used to select a discriminative set of concepts that maximise coverage of class semantics while minimizing redundancy. Specifically, we define a set function $f(S) = alpha dot "coverage"(S) - beta dot "redundancy"(S)$, and select the subset $S subset.eq C$ of concepts by approximately maximizing $f(S)$ via a greedy algorithm. As an improvement to the original paper's algorithm:

1. We incorporate CLIP + BioMedCLIP embeddings into the selection process to account for global similarity. This also implicitly makes sure that only textual concepts that are semantically understood by the VLM are part of the final selection. We modify this architecture for the mixture-of-experts scenario, doing expert-wise bottleneck maintenance.
2. Concepts are stemmed, filtered for stopwords, and pruned to remove any that contain morphological variants of class names — done to reduce semantic leakage and prevent the model from trivially associating concepts with their target classes.

This mechanism proves particularly valuable for advanced biomedical terminology — such as _telangiectasia_, _ovoid_ or _keratinization_ — which are well-represented in BioMedCLIP’s domain corpus but may not be meaningfully encoded by CLIP. By filtering concepts through this embedding-informed scoring process, we obtain a lean and discriminative concept set that adapts to each expert model. As seen in the example concept list, the generalist leans towards visual descriptors, while the specialist uses very specific terminology, whose visual context is implicitly encoded due to the training corpus #footnote[e.g. "Keratinization is defined as cytoplasmic events that take place"]. It is unsurprising that common words such as "presence", "brown", "areas" and "pigmentation" are widely used in both corpora #footnote[distributions computed using the nltk toolkits and the CLIP similarity scores]. 

#let divided(left, right) = {
  rect(
    stroke: 1pt,
    radius: 8pt,
    inset: 0pt,
    height: 3.8cm,
    grid(
      columns: (1fr, auto, 1fr),
      gutter: 2pt,
      pad(rest: 8pt, left),
      line(end: (0%, 100%)),
      pad(rest: 8pt, right)
    )
  )
}

#divided([
  *Global Specialist Concepts:*
  - Keratinization patterns
  - Erythematous base
  - Focal Nodularity
  - Multilobular pattern
  ...
], [
  *Global Generalist Concepts:*
  - Crusty texture
  - Small diameter
  - Pink
  - Light brown
  ...
])

During MoE, we freeze the concept selection bottlenecks, and use those selected during their corresponding uni-expert training cycles. This allows for fair comparisons and ensures that the data distribution is the only independent factor. 

// This is particularly useful for sophisticated biomedical nomenclature such as "Telangiectasia", "Ovoid", that are part of BioMedCLIP's training corpus but not that of CLIP's. This scoring process is not perfect, but we find that this process effectively reduces the bottleneck size, producing an optimal bottleneck for each model variation. During mixture-of-experts training, we freeze the concept selection module and load features selected during uni-expert training. 

== Mixture-of-Experts

#figure(
  image("images/bottled_brilliance_moe.png", width: 100%),
  caption: [Bottled Brilliance Neural Architecture. Additions to LaBO are highlighted in yellow.]
) <gating_dist>

=== The Case for Expertise

As seen in the selection process, some lesions are distinguishable by general visual attributes like colour patterns and asymmetry, which CLIP catures well #cite(<CLIP>). However, we hypothesise that there may be useful biomedical descriptors, that help when attempting to interpret decisions. For instance, a "multilobular pattern" has a distinct morphological shape, that when presented to an ordinary person, would appear obscure, but could highlight a clear analogy to a medical professional who would assess that analogy to build a deeper understanding of the diagnosis. The elucidated scenario comprises of a specialist audience, and a "common man" generalist. When representing this neurally, we find that it is reminiscent of the Mixture-of-Experts architecture #cite(<eigen2014learningfactoredrepresentationsdeep>). 

=== The Experts

CLIP is the choice architecture in LaBO; it learns a transferable visual representation by contrastively training image and text encoders on 400 million (image, text) pairs #cite(<CLIP>). It's wide training corpus and general understanding of worldly knowledge makes it a suitable candidate for the generalist. BioMedCLIP is a multimodal biomedical foundation model, trained on PMC-15M, a dataset containing 15 million biomedical image-text pairs that are taken from scientific articles in PubMed Central (PMC). The corpus taxonomy includes dermatology photos, microscopy, histopathology and X-Rays. 

To ensure a fair comparison, we standardise the architecture by using ViT-B/16 for both experts, instead of the ViT-L/14 used in LaBO. While ViT-L/14 outperforms the base variant, large-scale BioMedCLIP models are not publicly available; however, scaling laws suggest that performance would improve proportionally by increasing transformer complexity #cite(<Zhai_2022_CVPR>).

=== Formulation

Our Gated Mixture-of-Experts approach combines similarity embeddings from both CLIP ($E_C$) and BioMedCLIP ($E_B$). Our approach uses precomputed image-to-concept dot products from each expert, and learns a concept-to-class association matrix for both. Formally, given an input image vector $x_i$, we pre-compute the generalist and specialist dot products based on their image and concept vectors:

$
  {D^(g) in.small RR^(B times m_g), D^(s) in.small RR^(B times m_s)}
$

where $m_g$ and $m_s$ denote the number of generalist and specialist concepts respectively. $A^(g) in.small RR^(K times m_g)$ and $A^(s) in.small RR^(K times m_s)$ are learnable association matrices that map concepts to class logits. To encourage semantically meaningful class–concept associations, the individual association matrices are initialised using language model priors by selecting the closest concepts to each class name in the CLIP embedding space.


#showybox(
  title: "",
  frame: (
    border-color: green,
    title-color: green.lighten(30%),
    body-color: green.lighten(95%),
    footer-color: green.lighten(80%)
  ),
  footer: ""
)[
 $ mat(
  1, 1, 0, 0, 0, 0;
  0, 0, 1, 0, 1, 0;
  0, 0, 0, 1, 0, 1;
) $

Weight initialisation for association matrix, where rows = classes, and columns = concepts.
]


Class-level predictions from each expert are computed as $S^(g) = D^(g) × (A^(g))^T$ and $S^(s) = D^(s) × (A^(s))^T$. 

The gating network, tuned to inhibit over-parametrisation, is a two-layer neural network with a _LeakyReLU_ activation and sigmoid output, defined as:

$
  g(x_i) = σ ( W_2 ( "LeakyReLU"( W_1 ( "LayerNorm"(x_i)))))
$

$g(x_i) in.small [0, 1]$ dynamically determines the cross-expert weighting for each input and produces a weighted combination:

$
  S_i = g(x_i) ⋅ S_i^(s) + (1 - g(x_i)) ⋅ S_i^(g)
$

The Gated MoE model is trained by minimizing a total loss that consists of a classification loss (cross-entropy for single-label and binary cross-entropy for multi-label) and a gate entropy loss, that encourages the gating network to avoid overly deterministic decisions and completely depend on one of the experts (by collapsing $g(x_i) arrow {0, 1}$). Additional regularisers are added to encourage prediction diversity (disincentivize model from collapsing similarity scores) and sparse concept-to-class activations #footnote[the original paper did not employ these losses in their final ablations, so we replicate those same decisions]. In the results section, we share an ablation between different loss combinations. 

$
  ℒ = "CrossEntropy"(S, y) + λ_{div} ⋅ ( - E_{i} [ "Var"_{k} (S_{i,k}) ] ) + λ_{"L1"} ⋅ ( ||A^(g)||_1 + ||A^(s)||_1 )
$

// Let $x in.small RR^d$ be the input image embedding, $C^(g) in.small RR^(m_g times d)$ be generalist concept embeddings, where there are $m_g$ concepts and $C^(s) in.small RR^(m_s times d)$ be specialist concept embeddings, where there are $m_s$ concepts. $A^(g) in.small RR^(K times m_g)$ and $A^(s) in.small RR^(K times m_s)$ are learnable association matrices between classes and generalist concepts. $g(x) in.small [0, 1]$ is the gating scalar produced by the gating network for an input $x$. The class-level predictions from each expert are computed:

// $
// S^(g) = D^(g) times (A^(g))^T
// $
// $
// S^(s) = D^(s) times (A^(s))^T
// $

// For the generalist branch, class prototypes ar

== Experimental Infrastructure

All experiments were run on a single NVIDIA L40S GPU in the Department of Computer Science's GPU server, while Weights and Biases is used for experimental tracking #cite(<wandb>). We run all few-shot models for a maximum of 5000 epochs and run tests based on the best validation performance #footnote[tuned to prevent overfitting where the training accuracy can quickly hit 100% due to under-parametrisation]. To tackle the cold start issue for noisy gate parameters under few-shot scenarios, a 500 epoch warm start is allowed, where the only trainable parameters are the gate.

When running the experiments for the first time, a bug in the mixture of experts code that combined 2 

= Results

== Loss Ablation

Primary hyperparameter tuning is done on HAM10000, with the best configurations immediately ported over to COVID-QU-Ex.

#figure(
  caption: [Shot-by-Shot Results],
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center, center, center, center, center, center, center),
    stroke: none,
    toprule,
    table.header(
      [*Variation*],
      table.cell(colspan: 6, align: center)[*Validation Accuracy*],
    ),
    midrule,
    [], [1-shot], [2-shot], [4-shot], [8-shot], [16-shot], [All],
    [MoE],
    [0.314], [0.464], [0.248], [*0.439*], [0.464], [*0.792*],
    [$"MoE"_"entropy"$ ($lambda = 0.2$)],
    [*0.482*], [*0.494*], [0.248], [0.346], [*0.539*], [0.786],
    botrule
  )
)

The use of our gating entropy loss provides performance boosts in three of five shots (with an average improvement of 6.19%), representative of its utility in stabilising gate estimates, with an average and discouraging the gating network from collapsing too early to a single expert. The weighting $lambda_"entropy"$ for the entropy loss component is set at 0.2, to avoid saturating the loss computation. 

== Fully Supervised Baselines

We first evaluate the models in the complete models in a fully supervised setting. The original paper uses `ViT-L/14` in their architecture. However, for the sake of fair comparisons, we 

#figure(
  caption: [Individual Model Performance Results],
  table(
    columns: 5,
    align: center,
    stroke: none,
    toprule,
    table.header(
      [*Model Variant*],
      [*Val. Acc. (\%)*],
      [*Val. Loss*],
      [*Test Acc. (\%)*],
      [*Test Loss*]
    ),
    midrule,
    [ViT-B/16], [79.1], [0.5501], [76.8], [0.6126],
    [BioMedCLIP], [77.3], [0.69656], [75.03], [0.7916],
    // [MoE], [79.2], [6.1445], [*77.21*], [0.8235],
    [MoE], [*79.6*], [4.442], [*78.61*], [0.714],
    // [BioMedCLIP], [0.7730], [0.70034], [0.73731], [0.72475],
    botrule
  ),
)<indiv-model>

Under fully supervised conditions, standard CLIP still outperforms BioMedCLIP, likely because ample supervision allows for a sufficiently comprehensive representation—lessening the advantage of specialized domain expertise. The best performance is attained by the Mixture of Experts (MoE), outperforming both uni-expert counterparts by $approx 4.66%$. However, the elevated loss suggests that, while the model achieves strong accuracy, its occasional errors are highly confident misclassifications — potentially exacerbated by the gating mechanism’s sharp routing decisions. When analysing the average gate distribution, we find that the gate's $g(x_i)$ begins to fluctuate before reaching a batch-wise average of 0.8 by the final training epoch. This shows increased dependence on the specialist in the majority of cases. 


== Skin Lesion Classification

#figure(
  caption: [Shot-by-Shot Results (in \%) - ],
  table(
    columns: 11,
    align: (left, center, center, center, center, center, center, center, center, center, center),
    stroke: none,
    toprule,
    table.header(
      [*Architecture*],
      table.cell(colspan: 5, align: center)[*Validation Accuracy*],
      table.cell(colspan: 5, align: center)[*Test Accuracy*]
    ),
    midrule,
    [],
    [1-shot], [2-shot], [4-shot], [8-shot], [16-shot],
    [1-shot], [2-shot], [4-shot], [8-shot], [16-shot],
    midrule,
    [G (PC)],
    [23.0], [35.9], [24.4], [34.5], [53.0],
    [22.4], [38.1], [23.4], [31.9], [52.6],
    // [G (OC)],
    // [33.5], [31.2], [30.8], [32.3], [53.0],
    // [30.9], [34.4], [29.2], [31.1], [**54.0**],
    [S],
    [25.0], [38.8], [*49.4*], [*63.0*], [52.8],
    [27.2], [40.0], [*48.8*], [*61.7*], [52.3],
    [MoE],
    [31.4], [46.4], [24.8], [43.9], [46.4],
    [28.1], [48.1], [27.3], [42.9], [45.1],
    [$"MoE"_"entropy"$],
    [*48.2*], [*49.4*], [24.8], [34.6], [*54.6*],
    [*45.8*], [*50.2*], [23.5], [34.2], [*52.8*],
    botrule
  )
) <few-shot-results>

The specialist expert outperforms its counterparts by a substantial margin in ultra-low-shot settings (1-2 shots), showing that domain-specific knowledge provides strong performance boosts when there is minimal labelled data. However, under high-supervision and full-supervision, the specialist advantage diminishes, with the generalist outperforming it as soon as there is enough data to learn broad visual features and enrich the association matrix.

Interestingly, the Mixture-of-Experts excels at ultra-low-shot scenarios, where MoE partial insights are fused from both experts to provide surprisingly impressive performance ($>70%$) improvement. It must however be noted that the weights learned by gate may be suboptimal when there isn't sufficient supervision to enrich weighting decisions; in early epochs, the average gate weight $g(x_i)$ edges wavers $0.4 arrow.l.r 0.6$ Under mid-range supervision, the results do not consistently favour MoE, since partial supervision is likely insufficient for the gate to converge on an optimal blend, adversely hurting performance instead.

A general observation is that the mixture-of-experts lacks sufficient supervision in few-shot settings to train a sufficiently rich gate, with the full benefits of combining experts only visible during full supervision. There is also near-random and unexplainable fluctuations in 4- and 8- shots. 

=== Intuition about Foundation Gates

One solution to this problem would be to train the gate on a tangentially similar task and port the weights, so the starting representation would have some intuition about which types of images would be suitable for which expert. However, this would require clear cross-task alignment, and there is little formal proof to support this conjecture. 

// The specialist expert outperforms it's counterparts by an unsurprisingly large margin in ultra-low-shot settings. While it was outperformed by the generalist under full supervision, it can be hypothesised that the utility of concept expertise is particularly useful when there is extremely little data supervision to learn a sufficiently rich representation on. 

// In few-shot scenarios, we make interesting observations about the fluctuations. In ultra-low-shot scenarios and high-shot scenarios (including under full supervision as seen in #ref(<indiv-model>)), the Mixture-of-Experts outperforms it's simpler counterparts. For the low-shot case, this could be attributed to the inherent randomness of the routing; since both experts perform similarly, this improvement could only be attributed to randomness. In the higher-shot scenario, we hypothesise that the improvement could be due to having sufficient supervision to learn a richer gate parameterisation. Nevertheless, in few-shot scenarios, the gating mechanism doesn't prove to be as effective as it is under full supervision for this reason.


// Val Accuracy: 89.883, Val std: 0.000
// /home/sn666/.conda/envs/labo/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no true nor predicted samples. Use `zero_division` parameter to control this behavior.
//   _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
// Test Accuracy = 88.101, Macro F1 = 0.092

// #figure(
//   caption: [Individual Model Results],
//   table(
//     columns: 6,
//     align: center,
//     stroke: none,
//     toprule,
//     table.header(
//       [*Dataset*],
//       [*Variant*],
//       [*Val Acc*],
//       [*Val Loss*],
//       [*Test Acc*],
//       [*Test Loss*]
//     ),
//     midrule,
//     [HAM10000], [ViT-B/16], [0.791], [0.55009], [0.76915], [0.61261],
//     [HAM10000], [ViT-L/14], [0.792], [0.61521], [0.79900], [0.61158],
//     [HAM10000], [BioMedCLIP], [0.773], [0.69656], [0.75025], [0.7916],
//     [HAM10000], [BioMedCLIP], [0.7730], [0.70034], [0.73731], [0.72475],
//     [HAM10000], [ViT-B/16 + BioMedCLIP], [sdf], [dfs], [0.7721], [0.8235],
//     botrule
//   ),
// ) <individual-model-results>

#let three-line-table = tablem.with(
  render: (columns: auto, ..args) => {
    table(
      columns: columns,
      stroke: none,
      align: center + horizon,
      table.hline(y: 0),
      table.hline(y: 1, stroke: .5pt),
      ..args,
      table.hline(),
    )
  }
)

== Gating Fluctuations

The gate provides fascinating insights into the inner workings of the model. Across 1- to 2-shot settings, there is a dramatic fluctuation in $g(x)$ as it lacks sufficient training examples to learn a stable preference, bouncing back between near-0 and near-1. In the mid-shot graphs (4-shot and 8-shot), the gate still oscillates in phases but displays a modestly smoother pattern -- though there are steep dips towards 0 or 1 at certain epochs; likely indicative of preference collapses. By 16-shot, the trend stabilises, with a surprising number of samples favouring the generalist over the specialist. Under fully supervised training, the network has enough labels to make more confident routing decisions, with the histogram staying relatively consistent across epochs. Unsurprisingly, the specialist is favoured when all training samples are considered. Overall, it seems clear that sufficient supervision is necessary for the gate to build a stable representation. 

#subpar.grid(
  figure(image("images/1-shot-gate.png"), caption: [
    1-shot
  ]), <a>,
  figure(image("images/2-shot-gate.png"), caption: [
   2-shot.
  ]), <b>,
  figure(image("images/4-shot-gate.png"), caption: [
    4-shot 
  ]), <c>,
  figure(image("images/8-shot-gate.png"), caption: [
    8-shot
  ]), <d>,
    figure(image("images/16-shot-gate.png"), caption: [
      16-shot
  ]), <e>,
    figure(image("images/all-shot-gate.png"), caption: [
      All-shot
  ]), <e>,

  columns: (1fr, 1fr),
  caption: [Gating Distribution Acros Different Shots],
  label: <full>,
)

== COVID-QU-Ex

#figure(
  caption: [Shot-by-Shot Results (in \%) for COVID-QU-Ex],
  table(
    columns: 13,
    align: (left, center, center, center, center, center, center, center, center, center, center),
    stroke: none,
    toprule,
    table.header(
      [*Arch.*],
      table.cell(colspan: 5, align: center)[*Validation Accuracy*],
      table.cell(colspan: 5, align: center)[*Test Accuracy*]
    ),
    midrule,
    [],
    [1-shot], [2-shot], [4-shot], [8-shot], [16-shot], [All],
    [1-shot], [2-shot], [4-shot], [8-shot], [16-shot], [All],
    midrule,
    [G], [44.4], [48.09], [46.94], [59.20], [64.93], [87.36], [43.96], [47.86], [45.81], [58.94], [63.57], [86.6],
    [S], [39.0], [40.48], [*69.65*], [*68.79*], [67.7], [89.93], [38.53], [41.06], [*69.41*], [*68.04*], [69.22], [89.89],
    // [MoE], [*51.2*], [*50.13*], [46.80], [54.65], [65.28], [87.92], [*50.72*], [*49.40*], [45.71], [55.35], [64.54], [87.95],
    [MoE], [*45.62*], [*65.87*], [51.76], [61.88], [*78.62*], [*93.51*], [*47.20*], [*63.69*], [50.06], [61.90], [*77.61*], [*93.67*],
    botrule
  )
) <covid-x-results>

In extreme low-shot cases (1-2 shots), MoE significantly outperforms both the generalist and the specialist, suggesting that fusing domain-specific cues help overcome the data scarcity. Furthermore, we observe in the gating histogram that there is an even balance between the generalist and specialist weights ($0.2 < g(x) <0.6$), suggesting that both experts are equally consulted in low-shot scenarios. However, at 4-8 shots, the specialist begins to outperform it's counterparts, eclipsing gains from the gated combination of both experts. By 16-shots, there is sufficient supervision for the gating network to converge on a better routing algorithm for each individual sample, with MoE taking the lead and outperforming the specialist by $approx 11%$. 

Under full supervision, MoE attains the highest accuracy of all, with an outstanding accuracy of 93.67%, consistent with the hypothesis that enough labelled data can produce remarkable synergies. The gating distribution histogram in #ref(<covidx-gating_dist>) shows a sparse distribution of gate values, with a significant number prioritising the specialist towards the end of training, reflecting the value of domain-specific representations for X-Ray data. 

#figure(
  image("images/covidx-full-supervision-hist.png", width: 70%),
  caption: [Fully Supervised Gating Distribution for Covid-X]
) <covidx-gating_dist>

The specialist is also consistently strong across all shots, with incremental improvements as more data is appended -- unlike in dermatoscopy where the generalist eventually catches up other full supervision. BioMedCLIP was trained with an extensive volume of X-Ray data ($10^6$ samples), an order of magnitude higher than dermatoscopic samples. Crucially, X-Ray classification often hinges on subtle clinical markers (e.g. faint opacities, lesions) which are difficult to identify on a purely visual or textural level, requiring domain-specific context and known prior connections in the scientific literature to help the model make accurate diagnoses.

//  the specialist outperforms the generalist in fully supervised settings, likely due to the extensive volume of X-Ray data ($10^6$ samples) used to train BioMedCLIP. 

// When compared to dermatoscopy, Chest X-Rays also see more incremental gains across different shots, with more benefits attained through fusing the experts. One explanation could be the subtle visual clinical markers (such as faint opacities or lesions) that are difficult to identify visually alone, so domain-specific context and known connections help the model synergise. However, any benefits that are a result of the generalist are likely faint, due to the strong performance of the specialist alone. 

= Conclusion and Limitations

In this work, we introduce a Mixture-of-Experts extension to concept bottleneck models, designed to fuse domain-specific knowledge with a more general visual understanding. The architecture learns a gate between experts to make weighting decisions, and is particularly remarkable under full-supervision, where there is sufficient training data to learn a strong gate representation. The hypotheses are validated across HAM10000 (dermatoscopy) and COVID-QU-Ex (chest X-rays). 

However, few-shot performance is still relatively unstable, despite regularisation methods, and it is notably difficult to sufficiently understand a dataset to efficiently synergise the experts. Future work can pre-train the gating network on related biomedical tasks (X-Ray Gate $arrow.l.r$ Dermatoscopy Gate), initialising it in our MoE framework with a general intuition for expert selection. Furthermore, due to computational constraints, the individual experts have their encoders frozen in the current architecture. An intriguing next step would be to allow partial fine-tuning or cross-expert transfer learning through residual connections #cite(<zhao2024hypermoebettermixtureexperts>), potentially further aligning the experts' representations while preserving their distinct domain-specific strengths. 

// Moreover, because the gating module itself requires enough data to robustly learn routing decisions, initialising it with a "foundation" gate (pre-trained across biomedical tasks) could build even deeper intuitions, assuming the tasks have strong connections. 


// A key limitation lies in the risk of implicit leakage when using technical jargon. Certain biomedical terms may be so specific that the model can effectively learn direct mappings from concept text to class labels. For instance, words like "metastatic" could immediately hint to a particular diagnosis, creating potential shortcuts that undermine fair evaluation of the model's grounding. While every effort was taken in this work to semantically prune the bottleneck, future work can explore more robust concept filtration methods, that involve a doctor's input. 



// We find that there is potential for implicit leakage when operating with technical jargon that a model can immediately connect to an actual diagnosis. 

// - Interestingly, there is potential for leakage, since the concept's actual semantic definition can be overly synonymous in certain cases. However, since the specialist expert did not outperform the generalist under the fully supervised setting, the leakage isn't immediately obvious.  

= Notes

- Using ViT-B/16 to establish the baseline in this paper - because it outputs 512 dimensions, which is the same as MedCLIP and BioMedCLIP
- Motivation - biomedical explainability is important - do more specialised variants do a better job
- We can't directly assess the quality of the explanation, but we can implicitly assess them through the expressiveness of the concept alignment
- Hypothesis - combine generalist + specialist improve interpretability and concepts
- Try initialising association weights using `gen_init_weight_from_cls_name` -- might be useful in few-shot scenario
- Some of the accuracies in the table were from the last epoch, not the best epoch - make sure to check

== Research Questions
1. First, does separating concept spaces improve interpretability and classification performance? 
2. Second, can learned fusion weights outperform naive averaging of similarity scores? 
3. And third, does the specialist model contribute more on rare or complex conditions?

== Methodology

- Run individual models - done
- Run BioMedCLIP with more specialised features
- Do hybrid gating between MedCLIP and BioMedCLIP - use different concept sets - only modify asso_opt.py file
- CLIP explainability - https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb#scrollTo=3ogYpvQAAH4s

If there's time:

- Linear probe NIH-XRay
- Apply best method from above

// = Results

// #figure(
//   caption: [Individual Model Results],
//   table(
//     columns: 8,
//     align: center,
//     stroke: none,
//     toprule,
//     table.header(
//       [*Dataset*],
//       [*Concept Set*],
//       [*Variant*],
//       [*Shot*],
//       [*Val Acc*],
//       [*Val Loss*],
//       [*Test Acc*],
//       [*Test Loss*]
//     ),
//     midrule,
//     [HAM10000], [Generalist], [ViT-B/16], [All], [0.791], [0.55009], [0.76915], [0.61261],
//     [HAM10000], [Generalist], [ViT-L/14], [All], [0.792], [0.61521], [0.79900], [0.61158],
//     [HAM10000], [Generalist], [BioMedCLIP], [All], [0.773], [0.69656], [0.75025], [0.7916],
//     [HAM10000], [Specialist], [BioMedCLIP], [All], [0.7730], [0.70034], [0.73731], [0.72475],
//     [HAM10000], [MoE], [ViT-B/16 + BioMedCLIP], [All], [sdf], [dfs], [0.7721], [0.8235],
//     botrule
//   ),
// ) <individual-model-results>

= 

// Add bibliography and create Bibiliography section
#bibliography("bibliography.bib")

// Create appendix section
#show: arkheion-appendices
=

== Prompt Generation


== Ablation Study on Mixture-of-Experts

// #figure(
//   caption: [Shot-by-Shot Results],
//   table(
//     columns: 7,
//     align: (left, center, center, center, center, center, center, center, center, center, center, center, center),
//     stroke: none,
//     toprule,
//     table.header(
//       [*Arch.*],
//       table.cell(colspan: 6, align: center)[*Validation Accuracy*],
//     ),
//     midrule,
//     [sdfsf], [2], [4], [8], [16], [All],
//     [1], [2], [4], [8], [16], [All],
//     midrule,
//     [G (PC)],
//     [0.230], [0.359], [0.244], [0.345], [0.546], [—],
    
//     botrule
//   )
// ) <moe-ablation>

// #figure(
//   caption: [Individual Model Results],
//   table(
//     columns: 5,
//     align: center,
//     stroke: none,
//     toprule,
//     table.header(
//       [*Variant*],
//       [*Val Acc*],
//       [*Val Loss*],
//       [*Test Acc*],
//       [*Test Loss*]
//     ),
//     midrule,
//     [ViT-B/16], [0.791], [0.55009], [0.76915], [0.61261],
//     [ViT-L/14], [0.792], [0.61521], [0.79900], [0.61158],
//     botrule
//   ),
// )
// 
// #figure(
//   caption: [Shot-by-Shot Results],
//   table(
//   columns: 14,
//   align: (left, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon, left + horizon),
//   stroke: none,
// table.cell(
//     align: horizon,
// )[Data],
// [Arch],
// table.cell(
//     colspan: 6,
//     align: left,
// )[Validation],
// table.cell(
//     colspan: 6,
//     align: left,
// )[Test],
// table.cell(
//     rowspan: 5,
//     align: horizon,
// )[HAM1],
// [],
// [1],
// [2],
// [4],
// [8],
// [16],
// [All],
// [1],
// [2],
// [4],
// [8],
// [16],
// [All],
// [Generalist (Paper's Concepts)],
// [0.23],
// [0.359],
// [0.244],
// [0.345],
// [0.546],
// [],
// [0.2239],
// [0.3811],
// [0.2338],
// [0.3194],
// [0.5284],
// [],
// [Generalist (Our Concept Generation)],
// [0.335],
// [0.312],
// [0.308],
// [0.323],
// [0.53],
// [0.81],
// [0.30945],
// [0.34428],
// [0.2915],
// [0.3114],
// [0.5403],
// [],
// [Specialist],
// [0.25],
// [0.388],
// [0.494],
// [0.63],
// [0.528],
// [],
// [0.2716],
// [0.4],
// [0.4876],
// [0.6169],
// [0.5234],
// [],
// [Mixture-of-Experts],
// [],
// [],
// [],
// [],
// [],
// [],
// [],
// [],
// [],
// [],
// [],
// [],
// )
// )

#figure(
  caption: [Shot-by-Shot Results],
  table(
    columns: 13,
    align: (left, center, center, center, center, center, center, center, center, center, center, center, center),
    stroke: none,
    toprule,
    table.header(
      [*Arch.*],
      table.cell(colspan: 6, align: center)[*Validation Accuracy*],
      table.cell(colspan: 6, align: center)[*Test Accuracy*]
    ),
    midrule,
    [],
    [1], [2], [4], [8t], [16], [All],
    [1], [2], [4], [8], [16], [All],
    midrule,
    [G (PC)],
    [0.230], [0.359], [0.244], [0.345], [*0.546*], [—],
    [0.2239], [0.3811], [0.2338], [0.3194], [*0.5284*], [—],
    [G (OC)],
    [0.335], [0.312], [0.308], [0.323], [0.530], [0.810],
    [0.3095], [0.3443], [0.2915], [0.3114], [0.540], [0.769],
    [S],
    [0.250], [0.388], [*0.494*], [*0.630*], [0.528], [—],
    [0.2716], [*0.40*], [*0.488*], [*0.617*], [0.5234], [0.7503],
    [MoE (st.)],
    [0.314], [0.464], [0.248], [0.439], [0.464], [0.792],
    [0.2806], [0.4806], [0.2726], [0.4289], [0.4508], [*0.772*],
    [MoE (st.)],
    [*0.482*], [*0.494*], [0.248], [0.346], [0.539], [0.786],
    [*0.458*], [*0.502*], [0.2348], [0.3423], [0.5114], [0.7592],
    botrule
  )
)
// 
// #figure(
//   caption: [Shot-by-Shot Results],
//   table(
//     columns: 11,
//     align: (left, center, center, center, center, center, center, center, center, center, center, center, center),
//     stroke: none,
//     toprule,
//     table.header(
//       [*Arch.*],
//       table.cell(colspan: 5, align: center)[*Validation Accuracy*],
//       table.cell(colspan: 5, align: center)[*Test Accuracy*]
//     ),
//     midrule,
//     [],
//     [1], [2], [4], [8], [16],
//     [1], [2], [4], [8], [16],
//     midrule,
//     [G (PC)],
//     [0.230], [0.359], [0.244], [0.345], [*0.546*],
//     [0.2239], [0.3811], [0.2338], [0.3194], [*0.5284*],
//     [G (OC)],
//     [0.335], [0.312], [0.308], [0.323], [0.530],
//     [0.3095], [0.3443], [0.2915], [0.3114], [0.540],
//     [S],
//     [0.250], [0.388], [*0.494*], [*0.630*], [0.528],
//     [0.2716], [*0.40*], [*0.488*], [*0.617*], [0.5234],
//     [MoE],
//     [0.314], [0.464], [0.248], [0.439], [0.464],
//     [0.2806], [0.4806], [0.2726], [0.4289], [0.4508],
//     [$"MoE"_"entropy"$],
//     [*0.482*], [*0.494*], [0.248], [0.346], [0.539],
//     [*0.458*], [*0.502*], [0.2348], [0.3423], [0.5114],
//     botrule
//   )
// )