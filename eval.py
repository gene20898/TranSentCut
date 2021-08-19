# %%
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import argparse
from utils import label_cut_noCut
from transformers import RobertaForSequenceClassification
from torch import nn
from transformers import CamembertTokenizer
from torch.utils.data import DataLoader
from typing import List
from torch.utils.data import Dataset
import re

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="the path to the model")
parser.add_argument("--tokenizer_path", type=str, required=True, help="the path to the tokenizer")
parser.add_argument("--eval_data_path", type=str, required=True, help="the path to the eval text file")
parser.add_argument("--context_length", type=int, required=True, help="the length of the context")
args = parser.parse_args()

# %%
model = RobertaForSequenceClassification.from_pretrained(args.model_path)

sm = nn.Softmax(dim=1)
tokenizer = CamembertTokenizer.from_pretrained(args.tokenizer_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
# dataset for inference
class SentDatasetInfer(Dataset):
    def __init__(self, data, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.data = data # list of [A,B]
        self.inputs = []
        for i, example in enumerate(self.data): 
            tok_result = self.tokenizer(
                example[0], # A sequence
                example[1], # B sequence
                max_length=args.context_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            tok_result['input_ids'] = torch.squeeze(tok_result['input_ids'])
            tok_result['attention_mask'] = torch.squeeze(tok_result['attention_mask'])
            self.inputs.append(tok_result)

    def __len__(self) -> int:   
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index]

# %%
def assemble_result(pred:list, text:str) -> list:
    # bookkeeping variables
    row = 0
    previous_space = 0
    sentence_ans = ""
    sentences_ans = []

    # find how many spaces in text
    start = 0
    pos = text.find(" ", start)
    contexts = []
    num2check = 0
    while(pos != -1):
        num2check += 1
        start = pos + 1
        pos = text.find(" ", start)

    for i, c in enumerate(text):
        if c == " " and pred[row] == 0: # no cut
            x = text[previous_space:i]
            previous_space = i+1
            sentence_ans += x + " "
            row += 1
        elif c == " " and pred[row] == 1: # cut
            x = text[previous_space:i]
            previous_space = i+1
            sentence_ans += x + " "
            sentences_ans.append(sentence_ans.strip())
            sentence_ans = ""
            row += 1

        if row == num2check and i == len(text) - 1: # for the last sentence
            x = text[previous_space:i+1]
            previous_space = i+1
            sentence_ans += x + " "
            sentences_ans.append(sentence_ans.strip())

    return sentences_ans

# %%
def cut_sent(text: str, model_name: str="") -> dict:
    text = re.sub("\n", "", text).strip()
    # build context
    start = 0
    pos = text.find(" ", start)
    contexts = []
    while(pos != -1):
        left_context = text[0:pos]
        right_context = text[pos+1:]
        contexts.append([left_context, right_context])
        start = pos + 1
        pos = text.find(" ", start)
    
    # create dataset
    infer_dataset = SentDatasetInfer(contexts, tokenizer)
    infer_loader = DataLoader(infer_dataset, batch_size=8)

    # run inference
    pred_lst = []
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            output = model.forward(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            logits = output.logits
            pred = torch.argmax(sm(logits),dim=1)
            pred_lst += pred.cpu().detach().numpy().tolist()

    # assemble the result
    sentences = assemble_result(pred_lst, text)


    return {"sentences": sentences}

# %%
def label_cut_noCut(sentences:List[str]) -> list:
    """
    input: `sentences` is list of string (each string is one sentence)
    returns: list of strings '<cut>' or '<no_cut>'
    """
    # label each space as cut/not cut
    cut_notCut_label = []
    for i, s in enumerate(sentences): # each sentence in label
        # the next code line does this for the sentence s:
        # 1. remove leading/trailing space
        # 2. split by the space
        # 3. find the length of the split, the number of spaces is this length-1, this the number of <no_cut> in this sentence
        var1 = ["<no_cut>" for _ in range( len(s.strip().split(" "))-1 ) ]
        if i < len(sentences)-1: # add a <cut> if not the last sentence of paragraph
            cut_notCut_label += var1 + ["<cut>"]
        else: # last sentence of paragraph, no need to cut
            cut_notCut_label += var1

    return cut_notCut_label

# %%
# parse the text file
with open(args.eval_data_path, "r") as f:
    lines = f.readlines()

paragraphs = [] # list of list
paragraph = [] # list of string
for line in lines:
    if line != "\n": # blank line
        paragraph.append(line.strip())
    else:
        paragraphs.append(paragraph)
        paragraph = []

# %%
transformer_output = []
for p in tqdm(paragraphs):
    text = " ".join(p)
    toks = cut_sent(text)["sentences"]
    transformer_output.append(toks)
    
# %%
cut_notCut_label_all_trans = []
cut_notCut_label_all_gt = []
num_skips = 0
for i, (sentences1,sentences2) in enumerate( zip(transformer_output,paragraphs) ):
    label_cut_noCut_tmp_trans = label_cut_noCut(sentences1)
    label_cut_noCut_tmp_gt = label_cut_noCut(sentences2)
    if len(label_cut_noCut_tmp_gt) != len(label_cut_noCut_tmp_trans):
        num_skips += 1
        continue
    cut_notCut_label_all_trans += label_cut_noCut_tmp_trans
    cut_notCut_label_all_gt += label_cut_noCut_tmp_gt

# %%
print(f1_score(cut_notCut_label_all_gt, cut_notCut_label_all_trans, average="macro"))

# %%
print(classification_report(cut_notCut_label_all_gt, cut_notCut_label_all_trans, digits=4))
for t in transformer_output:
    print(t)
    print("")
