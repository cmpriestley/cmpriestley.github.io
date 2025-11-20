#claude wrote this to create a markdown file with embedded images from the mistral ocr output
#there may be a better way to do this in the mistral docs but this works for now

#!/usr/bin/env python3
import json

def replace_images_in_markdown(markdown_text, image_data):
    """Replace image references with base64 embedded images."""
    for image_id, base64_data in image_data.items():
        if base64_data:
            # Replace markdown image reference with embedded base64
            markdown_text = markdown_text.replace(
                f'![{image_id}]({image_id})',
                f'![{image_id}]({base64_data})'
            )
    return markdown_text

# Read the JSON file
with open('ocr_output.json', 'r') as f:
    data = json.load(f)

markdowns = []

# Process each page
for page in data['pages']:
    # Extract images from page into a dictionary
    image_data = {}
    for img in page['images']:
        image_data[img['id']] = img.get('image_base64', '')

    # Replace image placeholders with actual base64 images
    page_markdown = replace_images_in_markdown(page['markdown'], image_data)
    markdowns.append(page_markdown)

# Combine all pages
combined_markdown = "\n\n".join(markdowns)

# Write to file
with open('Cosmology_Notes_2025.md', 'w') as f:
    f.write('# Cosmology Notes 2025\n\n')
    f.write('*Transcribed from handwritten notes using Mistral OCR*\n\n')
    f.write('---\n\n')
    f.write(combined_markdown)

print("✓ Markdown file created: Cosmology_Notes_2025.md")
print(f"✓ Total pages processed: {len(data['pages'])}")
print(f"✓ Total images embedded: {sum(len(p['images']) for p in data['pages'])}")
