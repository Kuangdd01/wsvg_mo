from transformers import BertModel, BertTokenizer
import torch.nn as nn

def load_bert(name: str, cache_dir=None):
    # 加载预训练的BERT模型和分词器
    bert_model = BertModel.from_pretrained(name,cache_dir=cache_dir)
    tokenizer = BertTokenizer.from_pretrained(name,cache_dir=cache_dir)
    return bert_model, tokenizer




if __name__ == "__main__":
    a,b = load_bert("bert-base-uncased")
    # print(a,b)
    text = "a young girl"
    encode_input = b(text, return_tensors='pt')
    # print(a(**encode_input).shape)
    output = a(**encode_input)
    print(output.last_hidden_state.shape)
    print(output.pooler_output.shape)