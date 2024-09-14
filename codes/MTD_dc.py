import os  
import torch 
import torch.nn.functional as F
from PIL import Image  
from transformers import BlipProcessor, BlipModel  # BLIP image-captioning processor and model
import torchxrayvision as xrv  # For X-ray image classification
import skimage  
import torchvision 
import csv  

# Main function to find matching files and process images and texts
def find_matching_files_and_process(txt_folder, img_folder1, img_folder2, output_csv):
    # Determine if a GPU is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Initialize the BLIP model and processor for image captioning
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

        # Initialize the XRayVision DenseNet model for X-ray classification
        xray_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)

        # Get all text (.txt) files in the provided text folder
        txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

        #IFT enhances LLM training, Med-PaLM
        # Open the output CSV file for writing results
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header row in the CSV
            csvwriter.writerow(['Instruction', 'Input', 'Output'])

            # Loop through each .txt file in the folder
            for txt_file in txt_files:
                # Extract the base name (without the .txt extension)
                base_name = os.path.splitext(txt_file)[0]

                # Look for corresponding image files with the same base name in both image folders
                search_pattern = f"{base_name}_"

                # Get matching files from both image folders
                matching_files1 = [f for f in os.listdir(img_folder1) if f.startswith(search_pattern)]
                matching_files2 = [f for f in os.listdir(img_folder2) if f.startswith(search_pattern)]

                # If matching files are found in both folders, process the images
                if matching_files1 and matching_files2:
                    print(f"Processing matching files for: {txt_file}")

                    # Use the first matching file in each folder as the target image
                    frontal_image_path = os.path.join(img_folder1, matching_files1[0])
                    lateral_image_path = os.path.join(img_folder2, matching_files2[0])

                    # Process the frontal X-ray image with BLIP
                    frontal_embedding = process_image(frontal_image_path, processor, model, device)

                    # Process the lateral X-ray image with BLIP
                    lateral_embedding = process_image(lateral_image_path, processor, model, device)

                    # Concatenate the BLIP embeddings from both images
                    blip_embedding = torch.cat((frontal_embedding, lateral_embedding), dim=0)

                    # Classify the frontal image using the XRayVision model
                    classification_result = classify_xray(frontal_image_path, xray_model, device)

                    # Create the combined input prompt using the embeddings and classification result
                    combined_input = create_combined_prompt(blip_embedding, classification_result)

                    # Read the content of the corresponding .txt file for output
                    with open(os.path.join(txt_folder, txt_file), 'r', encoding='utf-8') as f:
                        output_content = f.read().strip()

                    # Write the instruction, input, and output to the CSV file
                    # BCE prompting to fine tune the Text decoder
                    csvwriter.writerow([
                        "Analyze chest X-ray data. Provide:\n1. Generate a more detailed finding for normal human understanding and non-healthcare professionals.\n2. Strictly DO NOT ADVISE MEDICINES and PRACTICES.",
                        combined_input,
                        output_content
                    ])

        print(f"CSV file has been created: {output_csv}")

    except Exception as e:
        # Handle any errors that occur during the process
        print(f"An error occurred: {str(e)}")

# Function to process an image using BLIP and return its embedding
def process_image(image_path, processor, model, device):
    try:
        # Load the image and convert it to RGB (color)
        image = Image.open(image_path).convert('RGB')

        # Add a dummy text input, as the BLIP model expects both image and text
        dummy_text = "This is an image."

        # Preprocess the image and text using the BLIP processor
        inputs = processor(images=image, text=dummy_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Send inputs to GPU/CPU

        # Run the BLIP model to generate image embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the image embeddings from the BLIP model's output
        image_embeds = outputs.image_embeds

        # Add a dimension if the embedding is 2D (required for further processing)
        if len(image_embeds.shape) == 2:
            image_embeds = image_embeds.unsqueeze(0)

        # If there is a CLS token, remove it (common in Transformer models)
        if image_embeds.shape[1] > 576:
            image_embeds = image_embeds[:, 1:, :]

        # Reshape the embedding into a grid (24x24 in this case)
        grid_size = int((image_embeds.shape[1])**0.5)
        image_embeds = image_embeds.reshape(1, grid_size, grid_size, -1)

        # Interpolate the grid to a 128x128 size for uniformity
        interpolated_embeds = F.interpolate(image_embeds.permute(0, 3, 1, 2), size=(128, 128), mode='bilinear', align_corners=False)

        # Reshape to 128x128 and get final embeddings by averaging over the sequence dimension
        interpolated_embeds = interpolated_embeds.permute(0, 2, 3, 1).reshape(1, 128*128, -1)
        averaged_embeds = torch.mean(interpolated_embeds, dim=1)  # Shape: (1, 768)

        # Further downsample to a 128-dimensional embedding
        final_embeds = F.interpolate(averaged_embeds.unsqueeze(0), size=(128,), mode='linear', align_corners=False)
        final_embeds = final_embeds.squeeze(0)  # Shape: (1, 128)

        return final_embeds.cpu()

    except Exception as e:
        # Handle any errors that occur during image processing
        print(f"Error processing image {image_path}: {str(e)}")
        return torch.zeros(1, 128)  # Return a zero tensor in case of error

# Function to classify an X-ray image using the XRayVision model
def classify_xray(image_path, model, device):
    try:
        # Read and normalize the image for X-ray classification
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)  # Normalize 8-bit image to [-1024, 1024] range
        img = img.mean(2)[None, ...]  # Convert to grayscale (single color channel)
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img = torch.from_numpy(img).to(device)

        # Run the X-ray image through the DenseNet model
        with torch.no_grad():
            outputs = model(img[None, ...])

        # Map the model's output to pathology names and probabilities
        results = dict(zip(model.pathologies, outputs[0].cpu().detach().numpy()))
        return results

    except Exception as e:
        # Handle errors during classification
        print(f"Error classifying X-ray {image_path}: {str(e)}")
        return {}  # Return an empty dictionary in case of error

#BCE Prompting
def create_combined_prompt(blip_embedding, classification_result):
    # Convert BLIP embeddings to a readable string
    blip_str = ', '.join(map(str, blip_embedding.tolist()))

    # Format classification results as "Condition: Probability%" strings
    classification_str = '\n'.join([f"{condition}: {probability*100:.2f}" for condition, probability in classification_result.items()])

    # Combine everything into the desired prompt format
    combined_prompt = (
        f"BLIP Embeddings:\n[{blip_str}]\n\n"
        f"Classification Results (%):\n"
        f"{classification_str}\n"
        f"Context: AI radiologist assistant analyzing BLIP-processed chest X-ray.\n"
        f"Limit: 7 sentences."
    )

    return combined_prompt

# Example usage of the function
txt_folder = 'path_to_IU-XRay_datset_findings'
img_folder1 = 'Path_Frontal_XRay_image_split'
img_folder2 = 'Path_Lateral_XRay_image_split'
output_csv = 'path_to_save_generated_dataset.csv'

# Call the main function to find matching files and process them
find_matching_files_and_process(txt_folder, img_folder1, img_folder2, output_csv)
