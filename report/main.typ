#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices
#let toprule = table.hline(stroke: 0.08em)
#let botrule = toprule
#let midrule = table.hline(stroke: 0.05em)

#show: arkheion.with(
  title: "Bottled Brilliance: Mixture of Experts for Biomedical Explainability",
  authors: (
    (name: "Sidharrth Nagappan", email: "sn666@cam.ac.uk", affiliation: "University of Cambridge", orcid: "0000-0000-0000-0000"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: lorem(55),
  keywords: ("First keyword", "Second keyword", "etc."),
)
#set cite(style: "chicago-author-date")
#show link: underline

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

