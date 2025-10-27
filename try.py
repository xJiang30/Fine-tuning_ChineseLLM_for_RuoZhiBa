import torch
print(torch.cuda.is_available())  # 输出 True 表示 GPU 可用
print(torch.cuda.device_count())  # 检查 GPU 数量
print(torch.cuda.get_device_name(0))  # 查看 GPU 的名称（例如 RTX 4060）

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bert_score import BERTScorer

# 下载 nltk 数据包
nltk.download('punkt_tab')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-large-chinese")
model = AutoModelForSeq2SeqLM.from_pretrained("fnlp/bart-large-chinese")

# 加载测试集 JSON 文件
with open('./val_data/val_fold_6.json', 'r',encoding='utf-8') as f:
    test_data = json.load(f)

# 对每个测试样本进行 tokenization
def tokenize_test_data(test_data):
    examples = []
    for entry in test_data:
        # 自动收集所有以 "annotated_result" 开头的参考文本，个数不定
        references = [entry[key] for key in entry if key.startswith("annotated_result")]
        examples.append({
            "input_text": entry["original_data"],
            "reference_texts": references  # 动态收集所有参考解释
        })
    return examples

test_data_processed = tokenize_test_data(test_data)

# 初始化 BERTScorer
bert_scorer = BERTScorer(lang="zh", rescale_with_baseline=True)

# 遍历测试数据生成解释并计算 BLEU 和 BERTScore
total_bleu_score = 0
total_bert_score = 0
num_samples = len(test_data_processed)

for item in test_data_processed:
    # 对测试输入进行 tokenization
    inputs = tokenizer(item["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=128)

    # 设置 pad_token_id 和 attention_mask
    inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    # 将输入移动到模型所在设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 使用模型生成输出
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,  # 控制生成长度
        num_beams=10,  # 使用 Beam Search 增加生成多样性
        no_repeat_ngram_size=2,  # 防止生成重复
    )

    # 解码生成的解释
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reference_texts = item["reference_texts"]

    print(f"Input: {item['input_text']}")
    print(f"Generated explanations: {generated_text}")
    print(f"reference: {reference_texts}\n")

    # 将生成的文本和参考文本分词
    generated_tokens = nltk.word_tokenize(generated_text)
    reference_tokens = [nltk.word_tokenize(ref) for ref in reference_texts]  # 每个参考文本进行分词

    # 计算 BLEU 分数（支持多个参考文本）
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    print(f"BLEU score: {bleu_score}\n")
    
    # 累加 BLEU 分数
    total_bleu_score += bleu_score

    # 计算 BERTScore
    bert_scores = []
    for ref in reference_texts:
        # 逐个参考文本计算 BERTScore
        P, R, F1 = bert_scorer.score([generated_text], [ref])
        bert_scores.append(F1.mean().item())  # 获取 F1 分数

    # 获取所有参考文本的平均 BERTScore
    avg_bert_score = sum(bert_scores) / len(bert_scores)
    total_bert_score += avg_bert_score
    
    print(f"BERTScore: {avg_bert_score}\n")

# 输出平均 BLEU 分数和 BERTScore
average_bleu_score = total_bleu_score / num_samples
average_bert_score = total_bert_score / num_samples

print(f"average BLEU score: {average_bleu_score}")
print(f"average BERTScore: {average_bert_score}")