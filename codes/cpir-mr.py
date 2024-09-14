import google.generativeai as genai  # Import the Google Generative AI package
import os 
import re

# Function to split the chained-SMRs into sections based on patterns
def extract_sections(text):
    # Split the text wherever the pattern of '\n\n**d. SECTION_NAME:**' appears (for example, a section header)
    sections = re.split(r'\n\n\*\*\d+\. [A-Z]+:\*\*\s*', text)

    # If the first section is empty, remove it
    if sections[0] == '':
        sections = sections[1:]

    # Trim whitespace around each section
    sections = [section.strip() for section in sections]
    return sections

# Configure the Google Generative AI model using an API key stored in an environment variable
genai.configure(api_key=os.environ["API_KEY"])

# Load the generative model "gemini-1.5-flash"
model = genai.GenerativeModel("gemini-1.5-flash")

# Define the number of files to process in each batch, we used 20
# Experiment done on 5, 10, 20, 50, 100
batch_size = 20

# Set the output folder for storing the detailed findings, and ensure it exists
output_folder = "path_to_save_generated_findings"
os.makedirs(output_folder, exist_ok=True)

# Get the list of input files from the actual findings folder and sort them
input_files = sorted(os.listdir("path_to_original_findings_folder"))

# Total number of files to process
total_files = len(input_files)

# A list of ordinals used to label the detailed outputs (e.g., FIRST, SECOND, etc.) based on batch size
ordinals = ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "NINTH", "TENTH",
            "ELEVENTH", "TWELFTH", "THIRTEENTH", "FOURTEENTH", "FIFTEENTH", "SIXTEENTH", "SEVENTEENTH", "EIGHTEENTH", "NINETEENTH", "TWENTIETH"]

# Loop through the input files in batches of `batch_size`
for i in range(0, total_files, batch_size):
    # Get the current batch of files to process
    batch_files = input_files[i:i+batch_size]
    texts = []  # List to hold the text contents of each file in the batch

    # Read the contents of each file in the current batch
    # We preprocessed the findings from IU XRay and save those in a .txt file with the similar id under findings folder.
    # The reports with no findings available are skipped
    for file_name in batch_files:
        file_path = os.path.join("./findings", file_name)
        with open(file_path, 'r') as file:
            texts.append(file.read())  # Add the file contents to the `texts` list

    # CPIR-MR ---------------------------------
    prompt = "\n\n I have {} examples of original findings. Add your notions in place of XXXX. Strictly DO NOT SUGGEST MEDICINE, PRACTICES. \n\n Original Findings: \n".format(len(texts))

    # Add each finding to the prompt, numbered consecutively
    for j, text in enumerate(texts, 1):
        prompt += "{}. {} \n".format(j, text)

    # Instruct gemini to generate detailed findings for each original finding, outputting 7-line summaries
    # prompt chaining
    prompt += "\n For each finding, generate a more detailed finding for normal human understanding in 7 lines and output those in {} points ".format(len(texts))

    # Add ordinals (e.g., FIRST, SECOND, THIRD) to the prompt for each finding
    for j in range(1, len(texts) + 1):
        prompt += "{}. {}: (detailed output) ".format(j, ordinals[j-1])

    # End the prompt with a period, for better prompting , CoT enhancements
    prompt += "."

    result = model.generate_content(["\n\n", prompt])

    # Extract the SMRs
    extracted_sections = extract_sections(result.text)

    # Write each generated section to the corresponding output file
    for j, section in enumerate(extracted_sections):
        if j != 0:  # Ensure the index starts from 1 to match file indices
            if j <= len(batch_files):  # Ensure we're within the batch size
                # Construct the output file path
                output_path = os.path.join(output_folder, batch_files[j-1])

                # Write the detailed output to the corresponding file
                with open(output_path, 'w') as output_file:
                    output_file.write(section)

                # Print confirmation of the processed file
                print(f"Processed: {batch_files[j-1]}")

# Print a message when all files are processed
print("All files processed.")
