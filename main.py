from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# Load label map
with open("id2label.json", "r", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("final_best_model")
model = AutoModelForSequenceClassification.from_pretrained("final_best_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

app = FastAPI()

class Complaint(BaseModel):
    text: str
    threshold: float = 0.5

@app.post("/predict")
def predict(data: Complaint):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)
    predicted_indices = (probs > data.threshold).nonzero(as_tuple=True)[1].tolist()
    predicted_labels = [id2label[i] for i in predicted_indices]

    return {
        "complaint": data.text,
        "predicted_labels": predicted_labels,
        "probabilities": probs.cpu().numpy().tolist()
    }
