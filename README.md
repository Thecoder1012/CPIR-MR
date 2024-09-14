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
