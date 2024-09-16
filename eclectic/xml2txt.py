import os
import xml.etree.ElementTree as ET

def extract_findings(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract patient ID
    patient_id = root.find('uId').get('id')
    
    # Extract findings
    findings = root.find('.//AbstractText[@Label="FINDINGS"]')
    if findings is not None and findings.text is not None:
        return patient_id, findings.text
    return patient_id, None

def process_xml_files(xml_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each XML file in the folder
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            try:
                patient_id, findings = extract_findings(xml_path)
                
                # Skip if no findings are available
                if findings is None:
                    print(f"Skipped {filename} - No findings available")
                    continue
                
                # Create and write to text file
                txt_filename = f"{patient_id}.txt"
                txt_path = os.path.join(output_folder, txt_filename)
                with open(txt_path, 'w') as txt_file:
                    txt_file.write(findings)
                
                print(f"Processed {filename} -> {txt_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Usage
xml_folder = './ecgen-radiology'
output_folder = './findings'
process_xml_files(xml_folder, output_folder)