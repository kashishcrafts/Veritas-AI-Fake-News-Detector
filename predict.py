from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("saved_model")
model = DistilBertForSequenceClassification.from_pretrained("saved_model")

text = input("Enter news text: ")

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

print("Raw logits:", outputs.logits)

probs = torch.softmax(outputs.logits, dim=1)

print("Probabilities:", probs)

prediction = torch.argmax(outputs.logits, dim=1).item()

print("Prediction class:", prediction)

if prediction == 1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")