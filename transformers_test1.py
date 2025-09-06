# Import the Hugging Face transformers pipeline
# The pipeline API allows easy access to pretrained models for common NLP tasks
from transformers import pipeline

# Import pandas for displaying results in a nice tabular format
import pandas as pd

# Create a text classification pipeline
# By default, this will load a sentiment-analysis model (like 'distilbert-base-uncased-finetuned-sst-2-english')
# This pipeline takes a text input and predicts labels such as 'POSITIVE' or 'NEGATIVE' with confidence scores
classifier = pipeline("text-classification")

# Example text: a complaint email from a customer (fun reference to Transformers characters!)
text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# Run the classifier on the text
# The output is a list of dictionaries, each containing:
#   'label'  -> predicted class (e.g., 'POSITIVE' or 'NEGATIVE')
#   'score'  -> confidence of the prediction (0.0 to 1.0)
outputs = classifier(text)

# Convert the outputs into a pandas DataFrame for a more readable table format
# This is especially useful when running batch predictions or logging results
if __name__ == "__main__":
    outputs = classifier(text)
    print(pd.DataFrame(outputs))
