import os
import re
from tqdm import tqdm
import google.generativeai as genai

# Define constants for easier configuration and maintenance
BATCH_SIZE = 20
OUTPUT_FOLDER = "./detailed_findings"
KEYWORDS_FOLDER = "./keywords"
INPUT_FOLDER = "./findings"

# List of ordinal numbers for generating prompts
ORDINALS = [
    "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "NINTH", "TENTH",
    "ELEVENTH", "TWELFTH", "THIRTEENTH", "FOURTEENTH", "FIFTEENTH", "SIXTEENTH", "SEVENTEENTH",
    "EIGHTEENTH", "NINETEENTH", "TWENTIETH"
]

# Function to extract sections from the generated text
def extract_sections(text):
    sections = re.split(r'\n\n\*\*\d+\. [A-Z]+:\*\*\s*', text)
    return [section.strip() for section in sections if section]

# Function to create necessary output folders
def setup_folders():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(KEYWORDS_FOLDER, exist_ok=True)

# Function to get a sorted list of input files
def get_input_files():
    return sorted(os.listdir(INPUT_FOLDER))

# Function to read the content of a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to generate the prompt for the Gemini model
def generate_prompt(texts):
    prompt = f"\n\nI have {len(texts)} examples of original findings. Add your notions in place of XXXX. Strictly DO NOT SUGGEST MEDICINE, PRACTICES.\n\nOriginal Findings:\n"
    prompt += "\n".join(f"{i}. {text}" for i, text in enumerate(texts, 1))
    prompt += f"\n\nFor each finding, generate a more detailed finding for normal human understanding in 7 lines and output those in {len(texts)} points "
    prompt += " ".join(f"{i}. {ORDINALS[i-1]}: (detailed output)" for i in range(1, len(texts) + 1))
    prompt += "."
    return prompt

# Function to generate the analysis prompt for the AI model
def generate_analysis_prompt(texts, detailed_findings):
    prompt = "\n\nFor each pair of original and detailed findings, provide the following analysis:\n"
    prompt += "1. List the 5 most important keywords from the original finding.\n"
    prompt += "2. List the 5 most important keywords from the detailed finding.\n"
    prompt += "3. Rate the similarity on a scale of 1-10, where 10 is perfectly similar.\n"
    prompt += "4. Briefly explain why you gave this similarity rating.\n\n"
    
    for i, (original, detailed) in enumerate(zip(texts, detailed_findings), 1):
        prompt += f"Original {i}: {original}\n\n"
        prompt += f"Detailed {i}: {detailed}\n\n"
    
    prompt += "Provide your analysis for each pair in the following format:\n"
    prompt += "Pair X:\n1. Original Keywords: [list]\n2. Detailed Keywords: [list]\n3. Similarity Rating: [1-10]\n4. Explanation: [brief explanation]\n\n"
    return prompt

# Function to process a batch of files
def process_batch(model, batch_files):
    # Read the content of each file in the batch
    texts = [read_file(os.path.join(INPUT_FOLDER, file)) for file in batch_files]
    
    # CPIR-MR -----
    # Generate detailed findings for the batch
    prompt = generate_prompt(texts)
    result = model.generate_content(["\n\n", prompt])
    extracted_sections = extract_sections(result.text)
    
    # CPMK-E ------
    # Generate analysis for the original and detailed findings
    analysis_prompt = generate_analysis_prompt(texts, extracted_sections[1:])
    analysis_result = model.generate_content(["\n\n", analysis_prompt])
    analysis_sections = analysis_result.text.split("Pair ")
    
    return extracted_sections, analysis_sections

# Function to write output to a file
def write_output(output_folder, file_name, content):
    with open(os.path.join(output_folder, file_name), 'w') as output_file:
        output_file.write(content)

# Main function to orchestrate the entire process
def main():
    # Configure the AI model
    genai.configure(api_key=os.environ["API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Set up the necessary folders
    setup_folders()
    
    # Get the list of input files
    input_files = get_input_files()
    total_files = len(input_files)
    
    # Process files in batches
    for i in tqdm(range(0, total_files, BATCH_SIZE), desc="Processing batches"):
        batch_files = input_files[i:i+BATCH_SIZE]
        extracted_sections, analysis_sections = process_batch(model, batch_files)
        
        # Write the results to output files
        for j, (file_name, section) in enumerate(zip(batch_files, extracted_sections[1:]), 1):
            write_output(OUTPUT_FOLDER, file_name, section)
            
            if j < len(analysis_sections):
                write_output(KEYWORDS_FOLDER, file_name, analysis_sections[j])
            
            print(f"Processed: {file_name}")
    
    print("All files processed.")

# Entry point of the script
if __name__ == "__main__":
    main()
