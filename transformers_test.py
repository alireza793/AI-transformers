from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("I love learning AI with Python!"))
print(classifier("This project is very hard and frustrating."))
