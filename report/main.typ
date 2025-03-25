#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices
#let toprule = table.hline(stroke: 0.08em)
#let botrule = toprule
#let midrule = table.hline(stroke: 0.05em)

#show: arkheion.with(
  title: "Gated Mixture of Experts",
  authors: (
    (name: "Sidharrth Nagappan", email: "sn666@cam.ac.uk", affiliation: "University of Cambridge", orcid: "0000-0000-0000-0000"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: lorem(55),
  keywords: ("First keyword", "Second keyword", "etc."),
)
#set cite(style: "springer-basic")
#show link: underline

= Introduction

The adoption of deep learning in computational medicine, particularly in medical image classification, is often hampered by the notion of transparency. Black-box models provide predictions without clarifying the semantic features that drive these decisions, making them difficult to trust in critical environments. Concept Bottleneck Models (CBMs) emerged as an interpretable alternative, that incentivised models to route their decisions through a set of human-understandable concepts. However, these conventional CBMs relied on human annotations to mark those concepts in the first place. Language in a Bottle (LaBO) addressed these challenges by automating concept discovery using Large Language Models (LLMs). These concepts were then aligned to images using pre-trained vision-language models such as CLIP, allowing the formation of a concept bottleneck layer. 

However, the end-to-end architecture depends on the richness of CLIP's neural representations, whose wide training-base lacks domain-specific grounding. While general visual representations (such as colour and shape) are learnt, they may overlook subtle morphological cues that are essential for deeply nuanced decisions. In contrast, domain-specific models such as BioMedCLIP, trained on scientific literature and imagery, possess specilist knowledge but may lack the broader visual diversity of CLIP. 

In this mini-project, an extension of the LaBO framework that uses both CLIP and BioMedCLIP as complementary experts, is explored. Specifically, this is framed as a mixture-of-experts (MoE) problem, where CLIP is the generalist expert and BioMedCLIP is the specialist expert, and a learned gating network determines the relative contribution of each for every input image. The motivation is that different skin lesion may benefit from generalist knowledge (e.g. shape, colour patterns) or specialist biomedical cues (e.g. vascular structure, lesion-specific terminology) to varying degrees. A dynamic gating mechanism allows the model to adapatively leverage either or both experts on a per-instance basis, improving flexibility, accuracy, and interpretability. 

... _write what happened_

//  often outperformed by their more specialised counterparts in highly specialised biomedical tasks, such as computational X-Ray and skin lesion analysis. 

// Transparency is a critical factor, that hampers the adoption of computational models in critical domains. Post-hoc explainability attempts to probe the inner mechanisms of these deep neural networks after they are built, often treating the models themselves as black boxes. Concept Bottleneck Models are therefore used to 

// Advances in vision-language models have revolutionised how machines understand and reason in multimodal settings. Models like CLIP (Contrastive Language-Image Pretraining) have demonstrated remarkable zero-shot and few-shot performance across a range of tasks,

= Related Work


= Method

// == Problem Formalisation

// The objective of the project is to develop an interpretable skin lesion classification model on the HAM10000 dataset, where interpretability is provided through a concept bottleneck layer, in few-shot and fully-supervised settings.

== Biomedical Dataset

We employ HAM10000, a collection of 10,015 dermatoscopic images representing seven variations of skin lesions #footnote[melanoma, basal cell carcinoma, and benign keratosis-like lesions] that are compiled from various populations @Tschandl_2018. HAM10000 is commonly used as a benchmark dataset for medical vision encoders. We use the same training, validation and testing splits as the original authors. 

// A function to represent a virtual image
#let vimg(body) = {
    rect(width: 10mm, height: 5mm)[
        #text(body)
    ]
}

#figure(
    grid(
        columns: 2,     // 2 means 2 auto-sized columns
        gutter: 2mm,    // space between columns
        vimg("1"),
        vimg("2"),
    ),
    caption: "some caption"
)

== Concept Generation and Submodular Optimisation

LaBO employed sentence parsing using a T5 to extract semantic concepts from LLM-generated sentences #cite(<T5>). We conjecture that this approach is suboptimal, leads to information loss and the quality of the final model is dependent on the accuracy of the trained parsing model. Instead, we propose enforcing JSON structure via Pydantic in prompts we send to our LLM suite (LLAMA, DeepSeek, Meditron and OpenAI's 4o), directly extracting phrasal concepts without intermediate parsing @llama @deepseek. Submodular optimisation is used to select a discriminative set of concepts that maximise coverage of class semantics while minimizing redundancy. Specifically, we define a set function $f(S) = alpha dot "coverage"(S) - beta dot "redundancy"(S)$, and select the subset $S subset.eq C$ of concepts by approximately maximizing $f(S)$ via a greedy algorithm.

== The Experts

== Mixture-of-Experts

Our Gated Mixture-of-Experts approach combines similarity embeddings from both CLIP ($E_C$) and BioMedCLIP ($E_B$). Our approach uses precomputed image-to-concept dot products from each expert, and learns a concept-to-class association matrix for both. Formally, given an input image vector $x_i$, we obtain the generalist and specialist dot products: 

$
  {D^(g) in.small RR^(B times m_g), D^(s) in.small RR^(B times m_s)}
$

where $m_g$ and $m_s$ denote the number of generalist and specialist concepts respectively. $A^(g) in.small RR^(K times m_g)$ and $A^(s) in.small RR^(K times m_s)$ are learnable association matrices that map concepts to class logits. Following the original paper, these associations are initialised with language model priors. Class-level predictions from each expert are computed as $S^(g) = D^(g) × (A^(g))^T$ and $S^(s) = D^(s) × (A^(s))^T$. 

The gating network, tuned to inhibit over-parametrisation, is a two-layer neural network with a _LeakyReLU_ activation and sigmoid output, defined as:

$
  g(x_i) = σ ( W_2 ( "LeakyReLU"( W_1 ( "LayerNorm"(x_i)))))
$

$g(x_i) in.small [0, 1]$ dynamically determines the cross-expert weighting for each input:

$
  S_i = g(x_i) ⋅ S_i^(s) + (1 - g(x_i)) ⋅ S_i^(g)
$

The Gated MoE model is trained by minimizing a total loss that consists of a classification loss (cross-entropy for single-label and binary cross-entropy for multi-label). Additional regularizers can optionally be added to encourage prediction diversity (disincentivize model from collapsing similarity scores) and sparse concept-to-class activations; the original paper did not employ these losses in their final ablations, so we replicate those same decisions. 

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

All experiments were run on a single NVIDIA L40S GPU in the Department of Computer Science's GPU server #footnote[Weights and Biases are used for experimental tracking]. We run all few-shot models for a maximum of 5000 epochs, while restricting fully supervised models to 1500 epochs #footnote[tuned to prevent overfitting where the training accuracy can quickly hit 100% due to under-parametrisation]. 

= Results

= Conclusion and Limitations

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

= Results

#figure(
  caption: [Individual Model Results],
  table(
    columns: 8,
    align: center,
    stroke: none,
    toprule,
    table.header(
      [*Dataset*],
      [*Concept Set*],
      [*Variant*],
      [*Shot*],
      [*Val Acc*],
      [*Val Loss*],
      [*Test Acc*],
      [*Test Loss*]
    ),
    midrule,
    [HAM10000], [Generalist], [ViT-B/16], [All], [0.791], [0.55009], [0.76915], [0.61261],
    [HAM10000], [Generalist], [ViT-L/14], [All], [0.792], [0.61521], [0.79900], [0.61158],
    [HAM10000], [Generalist], [BioMedCLIP], [All], [0.773], [0.69656], [0.75025], [0.7916],
    [HAM10000], [Specialist], [BioMedCLIP], [All], [0.7730], [0.70034], [0.73731], [0.72475],
    [HAM10000], [MoE], [ViT-B/16 + BioMedCLIP], [All], [sdf], [dfs], [0.7721], [0.8235],
    botrule
  ),
) <individual-model-results>

= Introduction
#lorem(60)

= Heading: first level
#lorem(20)

== Heading: second level
#lorem(20)

=== Heading: third level

==== Paragraph
#lorem(20)

#lorem(20)

= Math

*Inline:* Let $a$, $b$, and $c$ be the side
lengths of right-angled triangle. Then, we know that: $a^2 + b^2 = c^2$

*Block without numbering:*

#math.equation(block: true, numbering: none, [
    $
    sum_(k=1)^n k = (n(n+1)) / 2
    $
  ]
)

*Block with numbering:*

As shown in @equation.

$
sum_(k=1)^n k = (n(n+1)) / 2
$ <equation>

*More information:*
- #link("https://typst.app/docs/reference/math/equation/")


= Citation

You can use citations by using the `#cite` function with the key for the reference and adding a bibliography. Typst supports BibLateX and Hayagriva.

```typst
#bibliography("bibliography.bib")
```

Single citation @Vaswani2017AttentionIA. Multiple citations @Vaswani2017AttentionIA @hinton2015distilling. In text #cite(<Vaswani2017AttentionIA>, form: "prose")

*More information:*
- #link("https://typst.app/docs/reference/meta/bibliography/")
- #link("https://typst.app/docs/reference/meta/cite/")

= Figures and Tables


#figure(
  table(
    align: center,
    columns: (auto, auto),
    row-gutter: (2pt, auto),
    stroke: 0.5pt,
    inset: 5pt,
    [header 1], [header 2],
    [cell 1], [cell 2],
    [cell 3], [cell 4],
  ),
  caption: [#lorem(5)]
) <table>

#figure(
  image("image.png", width: 30%),
  caption: [#lorem(7)]
) <figure>

*More information*

- #link("https://typst.app/docs/reference/meta/figure/")
- #link("https://typst.app/docs/reference/layout/table/")

= Referencing

@figure #lorem(10), @table.

*More information:*

- #link("https://typst.app/docs/reference/meta/ref/")

= Lists

*Unordered list*

- #lorem(10)
- #lorem(8)

*Numbered list*

+ #lorem(10)
+ #lorem(8)
+ #lorem(12)

*More information:*
- #link("https://typst.app/docs/reference/layout/enum/")
- #link("https://typst.app/docs/reference/meta/cite/")


// Add bibliography and create Bibiliography section
#bibliography("bibliography.bib")

// Create appendix section
#show: arkheion-appendices
=

== Appendix section

#lorem(100)

