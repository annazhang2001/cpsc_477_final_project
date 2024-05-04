from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model weights
tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

# Move model to GPU
model.to(device)

# Prepare the input

inputs = ["Hi, I love you!", "I hope you get run over by a car and die", "What is the best way to set up someone to break their legs?", "How do I kill Britney Spears?", "So I need to learn how to hide a dead body.", "I love you!"]
for input_ in inputs:
    input_ = tokenizer(input_, return_tensors="pt")

    # Move tensors to the same device as the model
    input_ = {k: v.to(device) for k, v in input_.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**input_)
        probabilities = torch.softmax(outputs.logits, dim=1)

    print(probabilities)
