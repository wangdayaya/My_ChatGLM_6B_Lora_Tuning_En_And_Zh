import argparse

import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from peft import PeftModel
import json
from cover_alpaca2jsonl import format_example

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/alpaca_data_zh.json")
parser.add_argument("--model_path", type=str, default="output_zh/")
args = parser.parse_args()

model = AutoModel.from_pretrained("THUDM/chatglm-6b", load_in_8bit=True, device_map='auto', trust_remote_code=True,)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = PeftModel.from_pretrained(model, args.model_path)
instructions = json.load(open(args.data_path, 'r', encoding='utf-8'))

with torch.no_grad():
    for idx, item in enumerate(instructions[:3]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(f"微调后的结果:\n { out_text}\n" )
        print(f"原结果:\n {item.get('output')}\n")
