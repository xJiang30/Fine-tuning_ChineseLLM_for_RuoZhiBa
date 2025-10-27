import json
from datasets import Dataset
import torch

# Check memory allocated to PyTorch
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")

# Check free memory of PyTorch
print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")

# relief 
torch.cuda.empty_cache()

# check
print("After emptying cache:")
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")

# load JSON file
with open('./train_data/train_fold_1.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

# preprocession
def preprocess_data(data):
    examples = []
    for entry in data:
        for key in entry:
            if key.startswith("annotated_result"):
                examples.append({
                    "input_text": entry["original_data"],
                    "target_text": entry[key]
                })
    return examples

train_data = preprocess_data(data)

# transfer to Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    "input_text": [item["input_text"] for item in train_data],
    "target_text": [item["target_text"] for item in train_data]
})

print(train_dataset)

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-large-chinese")
model = AutoModelForSeq2SeqLM.from_pretrained("fnlp/bart-large-chinese")

# pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# tokenization
def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# map to whole dataset
tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)

from transformers import Trainer, TrainingArguments

# training parameter
training_args = TrainingArguments(
    output_dir="./results_bart_1", 
    evaluation_strategy="no",  # ban evaluation 
    learning_rate=3e-5, 
    per_device_train_batch_size=4,  # batch size
    per_device_eval_batch_size=4, 
    num_train_epochs=3, 
    weight_decay=0.01, 
    logging_dir="./logs",
    logging_steps=10,  
)

# define Trainer
trainer = Trainer(
    model=model, 
    args=training_args,  
    train_dataset=tokenized_datasets,  
)

# start train
trainer.train()

# save fine_tuned model and tokenizer
model.save_pretrained("./fine_tuned_bart_1")
tokenizer.save_pretrained("./fine_tuned_bart_1")

# try 
test_input = "在春天栽了一个跟头的李老汉，到了秋天跟头一个接一个"

# Tokenize，generate attention mask
inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

#cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

# generate strategy
outputs = model.generate(
    inputs["input_ids"], 
    attention_mask=inputs["attention_mask"],
    max_length=100,
    num_beams=5,  # beam search
    no_repeat_ngram_size=2,  # avoid repeat
    # top_k=50,  # top-k
    # temperature=0.7,  # temperature
    # do_sample=True 
)

# try result
print("Generated explanations:", tokenizer.decode(outputs[0], skip_special_tokens=True))
