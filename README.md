# Towards making Reports Non-Expert perceivable: A simplified LLM Prompting
```latex
\noindent\rule{\linewidth}{0.4pt}
`` I have \{n\} examples of original findings. Add your notions in place of XXXX. Strictly DO NOT SUGGEST MEDICINES and PRACTICES.
The prompt is constructed iteratively as follows:
$\text{prompt} \mathrel{+}= ``\{j\}. \{text\} \backslash n"$\footnote{$j$ is the index of the current iteration and $text$ represents the original findings from IU X-Ray dataset to be added.}
% This format allows for sequential numbering and newline separation of prompt components.}
% \textit{Note: }
\text{prompt}= For each finding, generate a more detailed fi nding for normal human understanding in 7 lines and output those in points
where $n = |\text{texts}|$. Then, for each $j \in [1, n]$:
$\text{prompt} \mathrel{+}= \text{``} \overbrace{j}^{\text{index}} \text{. }
\overbrace{\text{ordinal}_j}^{\text{ordinal number}} 
\text{: } \underbrace{\text{(detailed output)}}_{\text{placeholder}} \text{"}$ ''
\noindent\rule{\linewidth}{0.4pt}
```
## Usage

This LaTeX code can be used to generate a structured prompt for analyzing medical findings, ensuring a consistent format and detailed explanations without recommending specific treatments.

## Sample BCE Prompt

```
Instruction:
Analyze chest X-ray data. Provide:
1. Generate a more detailed finding for normal human understanding and non-healthcare 
professionals. 
2. Strictly DO NOT ADVISE MEDICINES and PRACTICES.

Input:
BLIP Embeddings:
[0.024061, 0.050350,  .. , -0.043655, 0.010471, -0.025682, 0.003035, 0.056423, -0.058437,
0.023345, 0.055599, .. , -0.024742, 0.001734, -0.017154, -0.005055, 0.067611, -0.047344]

Classification Results (%):
Consolidation: 17.85
Nodule: 12.28
Fracture: 11.82
Infiltration: 9.05
Atelectasis: 8.90
Cardiomegaly: 8.16
Lung Opacity: 8.16
Mass: 8.07
Enlarged Cardiomediastinum: 4.33
Emphysema: 3.50
Edema: 3.38
Pneumonia: 1.66
Effusion: 1.08
Fibrosis: 0.87
Pleural_Thickening: 0.51
Pneumothorax: 0.46
Hernia: 0.10
Lung Lesion: 0.08

Context: AI radiologist assistant analyzing BLIP-processed chest X-ray.
Limit: 7 sentences.
```
## Usage

This prompt can be used with large language models or AI systems capable of interpreting medical imaging data to generate human-readable explanations of chest X-ray findings. It's designed to provide informative, yet non-prescriptive, insights suitable for general understanding.

## CPMK-E Outline

```python
analysis_prompt = """
For each pair of original and detailed findings, provide the following analysis:
1. List the 5 most important keywords from the original finding.
2. List the 5 most important keywords from the detailed finding.
3. Rate the similarity on a scale of 1-10, where 10 is perfectly similar.
4. Briefly explain why you gave this similarity rating.
"""

for j in range(1, n+1):
    analysis_prompt += f"Original {j}: {original_j}\n\n"
    analysis_prompt += f"Detailed {j}: {detailed_j}\n\n"

analysis_prompt += """
Provide your analysis for each pair in the following format:
Pair X:
1. Original Keywords: [list]
2. Detailed Keywords: [list]
3. Similarity Rating: [1-10]
4. Explanation: [brief explanation]
"""
```

Where:
- `n = |texts| = |extracted_sections| - 1`
- `original_j ∈ texts`
- `detailed_j ∈ extracted_sections[1:]`

## Usage

This CPMK-E outline can be used to generate prompts for analyzing and comparing original medical findings with their detailed counterparts. It's particularly useful for:

1. Extracting key information from medical texts
2. Comparing the similarity of original and detailed findings
3. Ensuring consistency in medical reporting and analysis

The resulting prompt can be used with language models or AI systems capable of processing and analyzing medical text data.

---

![Statistical Plots](https://github.com/Thecoder1012/CPIR-MR/blob/main/assets/combined_analysis_plots_v2.png)

![Qualitative Comparison](https://github.com/Thecoder1012/CPIR-MR/blob/main/assets/wual_v3.png)
