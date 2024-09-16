# 🏥 Towards making Reports Non-Expert perceivable: A simplified LLM Prompting

## 📋 Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [LaTeX Code for Structured Prompts](#latex-code-for-structured-prompts)
- [Biomedical Condition Embedding (BCE) Prompt](#biomedical-condition-embedding-bce-prompt)
- [Chain Prompting for Medical Keyword Extraction (CPMK-E)](#chain-prompting-for-medical-keyword-extraction-cpmk-e)
- [Code Implementation](#code-implementation)
  - [CPIR-MR Implementation (cpir-mr.py)](#cpir-mr-implementation-cpir-mrpy)
  - [MTD Dataset Creation (MTD_dc.py)](#mtd-dataset-creation-mtd_dcpy)
  - [XML to Text Conversion (xml2txt.py)](#xml-to-text-conversion-xml2txtpy)
  - [Data Visualization (plotting.py)](#data-visualization-plottingpy)
- [Datasets](#datasets)
- [Statistical Analysis](#statistical-analysis)
- [Qualitative Comparison](#qualitative-comparison)
- [Getting Started](#getting-started)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## 🎯 Introduction

This project aims to make medical reports more understandable for non-experts using Large Language Model (LLM) prompting techniques.

## 📁 Project Structure

```
project_root/
│
├── code/
│   ├── cpir-mr.py
│   └── MTD_dc.py
│
├── dataset/
│   ├── IU XRay SMR.xlsx
│   └── MTD_Dataset.csv
│
├── eclectic/
│   ├── findings.zip
│   ├── gemini1.5flash.zip
│   ├── xml2txt.py
│   ├── plotting.py
│   └── combined_analysis_plots.png
│
└── assets/
    ├── combined_analysis_plots_v2.png
    └── supp_comparison.png
```

## 📝 LaTeX Code for Structured Prompts

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

### 🔧 Usage

This LaTeX code can be used to generate a structured prompt for analyzing medical findings, ensuring a consistent format and detailed explanations without recommending specific treatments.

## 🧬 Biomedical Condition Embedding (BCE) Prompt

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

### 🔧 Usage

This prompt can be used with large language models or AI systems capable of interpreting medical imaging data to generate human-readable explanations of chest X-ray findings. It's designed to provide informative, yet non-prescriptive, insights suitable for general understanding.

## 🔍 Chain Prompting for Medical Keyword Extraction (CPMK-E)

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

### 🔧 Usage

This CPMK-E outline can be used to generate prompts for analyzing and comparing original medical findings with their detailed counterparts. It's particularly useful for:

1. Extracting key information from medical texts
2. Comparing the similarity of original and detailed findings
3. Ensuring consistency in medical reporting and analysis

The resulting prompt can be used with language models or AI systems capable of processing and analyzing medical text data.

## 💻 Code Implementation

Our project includes several Python scripts that implement our novel approaches and data processing:

### CPIR-MR Implementation (cpir-mr.py)

This script implements the Chain Prompting for Improved Readability - Medical Reports (CPIR-MR) technique, our main contribution. Key features include:

- Utilizes the Google Generative AI package with the "gemini-1.5-flash" model.
- Processes original medical findings in batches to generate more detailed, human-readable reports.
- Implements prompt chaining and Chain of Thought (CoT) enhancements for improved output.
- Generates Simplified Medical Reports (SMRs) structured in 7-line summaries.

### MTD Dataset Creation (MTD_dc.py)

This script creates the Medical Text Dataset (MTD) used for training our multimodal text decoder. It incorporates Biomedical Condition Embedding (BCE) prompting. Key features include:

- Uses BLIP (Bootstrapping Language-Image Pre-training) for image captioning.
- Employs TorchXRayVision for X-ray image classification.
- Combines BLIP embeddings and classification results to create comprehensive prompts.
- Generates a CSV file with instructions, inputs, and outputs for training.

### XML to Text Conversion (xml2txt.py)

Located in the `eclectic` folder, this script is used to convert XML files from the IU X-Ray dataset into plain text format. Key features include:

- Extracts relevant information from XML files
- Converts structured XML data into a more accessible text format
- Outputs are stored in `findings.zip` within the `eclectic` folder

### Data Visualization (plotting.py)

Located in the `eclectic` folder, this script is responsible for generating visualizations of our results. Key features include:

- Creates statistical plots and graphs
- Visualizes comparisons between original and simplified medical reports
- Outputs include the plots found in the `assets` folder

## 📊 Datasets

### Main Datasets

Our project includes two important datasets in the `dataset` folder:

1. **IU XRay SMR.xlsx**

This file is an extension of the IU XRay dataset with CPIR-MR (Chain Prompting for Improved Readability - Medical Reports). It contains three columns:

- **Id**: Image ID from the original IU XRay dataset
- **Original Report**: The original report from the IU XRay dataset
- **CPIR-MR**: Simplified Medical Reports (SMRs) generated by our CPIR-MR prompting technique

This dataset demonstrates how our method transforms original medical reports into more understandable versions for non-experts.

2. **MTD_Dataset.csv**

The MTD (Medical Text Dataset) is used for fine-tuning Llama3.1 with our BCE (Biomedical Condition Embedding) Prompting technique. It includes:

- BLIP embeddings: Vector representations of medical images
- Classification results: Percentages for various medical conditions detected in the images

This dataset showcases the integration of image analysis (via BLIP embeddings) and medical condition classification in our prompting approach.

### Additional Data Files

Located in the `eclectic` folder:

1. **findings.zip**: Contains the text files derived directly from the IU X-Ray dataset using the `xml2txt.py` script. These files represent the original medical findings in a plain text format.

2. **gemini1.5flash.zip**: Contains the output generated by the `cpir-mr.py` script using the Gemini 1.5 Flash model. These files represent the simplified medical reports (SMRs) created by our CPIR-MR technique.

These datasets are crucial for understanding, replicating, and building upon our work in making medical reports more accessible to non-experts.

## 📊 Statistical Analysis

![Statistical Plots](https://github.com/Thecoder1012/CPIR-MR/blob/main/assets/combined_analysis_plots_v2.png)

These plots, generated using the `plotting.py` script, provide a visual representation of our statistical analysis, comparing various aspects of the original medical reports and our simplified versions.

## 🔬 Qualitative Comparison

![Qualitative Comparison](https://github.com/Thecoder1012/CPIR-MR/blob/main/assets/supp_comparison.png)

This image showcases a qualitative comparison between original medical reports and our simplified versions, demonstrating the effectiveness of our approach in making medical information more accessible to non-experts.

## 🚀 Getting Started

To use this project:

1. Clone the repository:
   ```
   git clone https://github.com/Thecoder1012/CPIR-MR.git
   ```
2. Set up your Google Generative AI API key as an environment variable
3. Use `xml2txt.py` to process the original IU X-Ray dataset if starting from XML files
4. Run `cpir-mr.py` to generate simplified medical reports
5. Use `MTD_dc.py` to create the Medical Text Dataset for further analysis or model training
6. Utilize `plotting.py` to visualize results and generate statistical plots

## 🤝 Contributing

We welcome contributions to improve this project. Please feel free to submit issues or pull requests.
