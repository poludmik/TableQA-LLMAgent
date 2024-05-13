from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image
import pandas as pd

processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

# read image from plots/test_CSV_file_gdp27.png path
image = Image.open("../plots/testing_api.png")


inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(predictions[0], skip_special_tokens=True))

data_string = processor.decode(predictions[0], skip_special_tokens=True)

# Replace the <0x0A> with actual newlines
data_string = data_string.replace("<0x0A>", "\n")

# Split the string by newlines to get individual lines
lines = data_string.split("\n")

# Extract the title (optional)
title = lines[0].split("|")[1].strip()

# Initialize a list to hold each row as a dictionary
data = []

# Start processing from the second line (index 1)
for line in lines[1:]:
    parts = line.split("|")
    if len(parts) == 2:  # Ensure the line is valid
        country = parts[0].strip()
        index = parts[1].strip()
        data.append({"Country": country, "Happiness Index": index})

# Create a DataFrame
df = pd.DataFrame(data)
print(df)