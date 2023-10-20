import json
import random
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

def cross_dataloader(dscorpus, json_file, csv_file, batch_size, shuffle=False, no_negs= 30):
    """
    This function builds train, val, test dataframe for training and evaluating cross-encoder
    """
    with open(json_file, 'r') as f:
        retrieved_sub_list = json.load(f)
    df = pd.read_csv(csv_file)
    data = []
    for i in range(len(df)):
        tokenized_question = df['tokenized_question'][i]
        retrieved_sub_ids = retrieved_sub_list[i]
        ans_ids = str(df['ans_id'][i])
        ans_ids = [int(x) for x in ans_ids.split(", ")]
        ans_sub_ids = df['ans_sub_id'][i][1:-1]
        ans_sub_ids = [int(x) for x in ans_sub_ids.split(", ")]
        for a_sub_id in ans_sub_ids:
            tokenized_text = dscorpus['tokenized_text'][a_sub_id]
            example = InputExample(texts=[tokenized_question, tokenized_text], label=0)
            for j in range(no_negs):
                data.append(example)
                
        neg_ids = [x for x in retrieved_sub_ids if dscorpus['id'][x] not in ans_ids]
        neg_ids = neg_ids[:no_negs]
        for neg_id in neg_ids:
            tokenized_text = dscorpus['tokenized_text'][neg_id]
            example = InputExample(texts=[tokenized_question, tokenized_text], label=1)
            data.append(example)        
    cross_dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)    
    return cross_dataloader
