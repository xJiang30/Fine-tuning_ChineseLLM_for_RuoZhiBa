from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bert_score import BERTScorer

# load nltk
nltk.download('punkt_tab')

# load fine_tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_bart_1")
model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_bart_1")

# load validation JSON file
with open('./val_data/val_fold_1.json', 'r',encoding='utf-8') as f:
    test_data = json.load(f)

# tokenization
def tokenize_test_data(test_data):
    examples = []
    for entry in test_data:
        # collect "annotated_result"
        references = [entry[key] for key in entry if key.startswith("annotated_result")]
        examples.append({
            "input_text": entry["original_data"],
            "reference_texts": references 
        })
    return examples

test_data_processed = tokenize_test_data(test_data)

# init BERTScorer
bert_scorer = BERTScorer(lang="zh", rescale_with_baseline=True)

#  BLEU & BERTScore
total_bleu_score = 0
total_bert_score = 0
num_samples = len(test_data_processed)

for item in test_data_processed:
    # tokenization
    inputs = tokenizer(item["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=128)

    # pad_token_id and attention_mask
    inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    # device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # generate strategy
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,  # 控制生成长度
        num_beams=10,  # 使用 Beam Search 增加生成多样性
        no_repeat_ngram_size=2,  # 防止生成重复
    )

    # decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reference_texts = item["reference_texts"]

    print(f"Input: {item['input_text']}")
    print(f"Generated explanations: {generated_text}")
    print(f"reference: {reference_texts}\n")

    # tokenization
    generated_tokens = nltk.word_tokenize(generated_text)
    reference_tokens = [nltk.word_tokenize(ref) for ref in reference_texts]

    # BLEU
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    print(f"BLEU score: {bleu_score}\n")
    
    # accumulate BLEU score
    total_bleu_score += bleu_score

    # BERTScore
    bert_scores = []
    for ref in reference_texts:
        P, R, F1 = bert_scorer.score([generated_text], [ref])
        bert_scores.append(F1.mean().item())  # F1

    # average BERTScore
    avg_bert_score = sum(bert_scores) / len(bert_scores)
    total_bert_score += avg_bert_score
    
    print(f"BERTScore: {avg_bert_score}\n")

average_bleu_score = total_bleu_score / num_samples
average_bert_score = total_bert_score / num_samples

print(f"average BLEU score: {average_bleu_score}")
print(f"average BERTScore: {average_bert_score}")
