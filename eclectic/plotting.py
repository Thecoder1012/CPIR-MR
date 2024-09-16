import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from textblob import TextBlob
from scipy import stats
import networkx as nx
from wordcloud import WordCloud
from matplotlib.gridspec import GridSpec

# Function definitions
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def extract_keywords(text):
    match = re.search(r'Keywords: ([\w\s,]+)', text)
    return [word.strip() for word in match.group(1).split(',')] if match else []

def analyze_keyword_overlap(original_keywords, detailed_keywords):
    original_set = set(original_keywords)
    detailed_set = set(detailed_keywords)
    overlap = original_set.intersection(detailed_set)
    return len(overlap) / len(original_set) * 100 if original_set else 0

def analyze_text_complexity(text):
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    return flesch_reading_ease, flesch_kincaid_grade

def get_important_keywords(text, n=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    important_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:n]
    return [keyword for keyword, _ in important_keywords]

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def extract_similarity_score(text):
    match = re.search(r'Similarity Rating: (\d+)', text)
    return int(match.group(1)) if match else None

def extract_medical_entities(text):
    medical_terms = ['cancer', 'tumor', 'lesion', 'fracture', 'inflammation']
    return [term for term in medical_terms if term in text.lower()]

def analyze_sentence_structure(text):
    if not isinstance(text, str):
        return 0
    text = re.sub(r'[^\w\s]', '', text)
    sentences = textstat.sentence_count(text)
    words = textstat.lexicon_count(text)
    return words / sentences if sentences > 0 else 0

def calculate_info_retention(row):
    original_grade = row['Original Flesch-Kincaid Grade']
    detailed_grade = row['Detailed Flesch-Kincaid Grade']
    return min(detailed_grade / original_grade, 1) if original_grade > 0 else 1

def adjust_list_length(lst, target_length, fill_value=None):
    if len(lst) < target_length:
        return lst + [fill_value] * (target_length - len(lst))
    return lst[:target_length]

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Main script
findings_folder = "./findings"
detailed_findings_folder = "./gemini1.5flash/detailed_findings"
keywords_folder = "./gemini1.5flash/keywords"
output_folder = "./analysis_results"
os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(findings_folder) if f.endswith('.txt')]

# Data collection
original_texts, detailed_texts, original_keywords, detailed_keywords = [], [], [], []
similarity_scores, overlap_percentages = [], []
original_complexity, detailed_complexity = [], []
original_important_keywords, detailed_important_keywords = [], []
original_sentiments, detailed_sentiments = [], []

for file in files:
    original_path = os.path.join(findings_folder, file)
    detailed_path = os.path.join(detailed_findings_folder, file)
    keywords_path = os.path.join(keywords_folder, file)

    original_text = read_file(original_path)
    detailed_text = read_file(detailed_path)
    keywords_text = read_file(keywords_path)

    # Skip this file if any of the required texts are missing
    if original_text is None or detailed_text is None or keywords_text is None:
        print(f"Skipping file {file} due to missing data")
        continue

    original_texts.append(original_text)
    detailed_texts.append(detailed_text)

    orig_keywords = extract_keywords(keywords_text)
    det_keywords = extract_keywords(keywords_text)
    original_keywords.append(orig_keywords)
    detailed_keywords.append(det_keywords)

    similarity_score = extract_similarity_score(keywords_text)
    if similarity_score is not None:
        similarity_scores.append(similarity_score)

    overlap_percentages.append(analyze_keyword_overlap(orig_keywords, det_keywords))
    
    original_complexity.append(analyze_text_complexity(original_text))
    detailed_complexity.append(analyze_text_complexity(detailed_text))

    original_important_keywords.append(get_important_keywords(original_text))
    detailed_important_keywords.append(get_important_keywords(detailed_text))

    original_sentiments.append(get_sentiment(original_text))
    detailed_sentiments.append(get_sentiment(detailed_text))

# Adjust list lengths
lengths = {
    'files': len(files),
    'original_texts': len(original_texts),
    'detailed_texts': len(detailed_texts),
    'original_keywords': len(original_keywords),
    'detailed_keywords': len(detailed_keywords),
    'similarity_scores': len(similarity_scores),
    'overlap_percentages': len(overlap_percentages),
    'original_complexity': len(original_complexity),
    'detailed_complexity': len(detailed_complexity),
    'original_important_keywords': len(original_important_keywords),
    'detailed_important_keywords': len(detailed_important_keywords),
    'original_sentiments': len(original_sentiments),
    'detailed_sentiments': len(detailed_sentiments)
}

most_common_length = Counter(lengths.values()).most_common(1)[0][0]

files = adjust_list_length(files, most_common_length)
original_texts = adjust_list_length(original_texts, most_common_length)
detailed_texts = adjust_list_length(detailed_texts, most_common_length)
original_keywords = adjust_list_length(original_keywords, most_common_length)
detailed_keywords = adjust_list_length(detailed_keywords, most_common_length)
similarity_scores = adjust_list_length(similarity_scores, most_common_length)
overlap_percentages = adjust_list_length(overlap_percentages, most_common_length)
original_complexity = adjust_list_length(original_complexity, most_common_length)
detailed_complexity = adjust_list_length(detailed_complexity, most_common_length)
original_important_keywords = adjust_list_length(original_important_keywords, most_common_length)
detailed_important_keywords = adjust_list_length(detailed_important_keywords, most_common_length)
original_sentiments = adjust_list_length(original_sentiments, most_common_length)
detailed_sentiments = adjust_list_length(detailed_sentiments, most_common_length)

# Create DataFrame
df = pd.DataFrame({
    'File': files,
    'Original Text': original_texts,
    'Detailed Text': detailed_texts,
    'Original Keywords': original_keywords,
    'Detailed Keywords': detailed_keywords,
    'Similarity Score': similarity_scores,
    'Keyword Overlap': overlap_percentages,
    'Original Complexity': original_complexity,
    'Detailed Complexity': detailed_complexity,
    'Original Important Keywords': original_important_keywords,
    'Detailed Important Keywords': detailed_important_keywords,
    'Original Sentiment': original_sentiments,
    'Detailed Sentiment': detailed_sentiments
})

# Separate complexity scores and calculate differences
df['Original Flesch Reading Ease'] = df['Original Complexity'].apply(lambda x: x[0])
df['Original Flesch-Kincaid Grade'] = df['Original Complexity'].apply(lambda x: x[1])
df['Detailed Flesch Reading Ease'] = df['Detailed Complexity'].apply(lambda x: x[0])
df['Detailed Flesch-Kincaid Grade'] = df['Detailed Complexity'].apply(lambda x: x[1])
df['Complexity Difference'] = df['Detailed Flesch Reading Ease'] - df['Original Flesch Reading Ease']

# Calculate Information Retention Score
df['Info Retention Score'] = df.apply(calculate_info_retention, axis=1)

# Add new columns
df['Original Medical Entities'] = df['Original Text'].apply(extract_medical_entities)
df['Detailed Medical Entities'] = df['Detailed Text'].apply(extract_medical_entities)
df['Original Sentence Complexity'] = df['Original Text'].apply(analyze_sentence_structure)
df['Detailed Sentence Complexity'] = df['Detailed Text'].apply(analyze_sentence_structure)

# Plotting
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2

fontsize = 18
fig, axs = plt.subplots(1, 5, figsize=(30, 6))

# 1. Distribution of Complexity Difference
sns.boxplot(y=df['Complexity Difference'], ax=axs[0])
axs[0].set_title('Complexity Difference\n(Detailed - Original)', fontsize=fontsize)
axs[0].set_ylabel('Flesch Reading Ease Difference', fontsize = fontsize)
axs[0].axhline(y=0, color='r', linestyle='--')

# 2. Sentiment Preservation
scatter = axs[1].scatter(df['Original Sentiment'], df['Detailed Sentiment'], 
                         c=df['Similarity Score'], s=df['Keyword Overlap'], cmap='coolwarm', alpha=0.7)
axs[1].set_title('Sentiment Preservation', fontsize=fontsize)
axs[1].set_xlabel('Original Sentiment', fontsize = fontsize)
axs[1].set_ylabel('Detailed Sentiment', fontsize = fontsize)
axs[1].plot([-1, 1], [-1, 1], 'r--', label='Perfect Preservation')
plt.colorbar(scatter, ax=axs[1], label='Similarity Score')

# 3. Distribution of Similarity Scores
sns.histplot(data=df, x='Similarity Score', kde=True, color='lightgreen', bins=10, ax=axs[2])
axs[2].set_title('Similarity Score Distribution', fontsize=fontsize)
axs[2].set_xlabel('Similarity Score', fontsize=fontsize)
axs[2].set_ylabel('Frequency', fontsize=fontsize)
axs[2].axvline(df['Similarity Score'].mean(), color='red', linestyle='--', label=f'Mean: {df["Similarity Score"].mean():.2f}')

# 4. Text Complexity Comparison
scatter = axs[3].scatter(df['Original Flesch Reading Ease'], df['Detailed Flesch Reading Ease'], 
                         c=df['Similarity Score'], s=df['Keyword Overlap'], cmap='viridis', alpha=0.7)
axs[3].set_title('Text Complexity Comparison', fontsize=fontsize)
axs[3].set_xlabel('Original Text Complexity', fontsize=fontsize)
axs[3].set_ylabel('Detailed Text Complexity', fontsize=fontsize)
axs[3].plot([0, 100], [0, 100], 'r--', label='Equal Complexity')
plt.colorbar(scatter, ax=axs[3], label='Similarity Score')

# 5. Enhanced Report Generation Performance
metrics = {
    'Avg Similarity': df['Similarity Score'].mean() / 10,
    'Avg Keyword Overlap': df['Keyword Overlap'].mean() / 100,
    'Readability Improvement': (df['Complexity Difference'].mean() / df['Original Flesch Reading Ease'].mean()) + 0.5,
    'Consistency': 1 - (df['Complexity Difference'].std() / df['Complexity Difference'].mean()),
    'Information Retention': df['Info Retention Score'].mean()
}

categories = list(metrics.keys())
values = list(metrics.values())

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
values = np.concatenate((values, [values[0]]))  # Repeat the first value to close the polygon
angles = np.concatenate((angles, [angles[0]]))  # Repeat the first angle to close the polygon

axs[4] = plt.subplot(155, polar=True)
axs[4].plot(angles, values)
axs[4].fill(angles, values, alpha=0.3)
axs[4].set_xticks(angles[:-1])
# axs[4].set_xticklabels(categories)
axs[4].set_title('Enhanced Report\nGeneration Performance', fontsize=fontsize)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'combined_analysis_plots.png'), dpi=600, bbox_inches='tight')
plt.close()

print("Combined analysis plots saved as 'combined_analysis_plots.png' in the output folder.")