import json
import random
import pandas as pd

def build_sub_cross_data(dscorpus, tokenizer, json_file, csv_file, no_negs= 30):
    """
    This function builds train, val, test dataframe for training and evaluating cross-encoder
    """
    sep_token = " . " + tokenizer.sep_token + " " 
    with open(json_file, 'r') as f:
        retrieved_sub_list = json.load(f)
    df = pd.read_csv(csv_file)    
    cross_texts, labels = [], []
    for i in range(len(df)):
        tokenized_question = df['tokenized_question'][i]
        retrieved_sub_ids = retrieved_sub_list[i]
        ans_ids = str(df['ans_id'][i])
        ans_ids = [int(x) for x in ans_ids.split(", ")]
        ans_sub_ids = df['ans_sub_id'][i][1:-1]
        ans_sub_ids = [int(x) for x in ans_sub_ids.split(", ")]
            
        for a_sub_id in ans_sub_ids:
            tokenized_text = dscorpus['tokenized_text'][a_sub_id]
            cross_text = tokenized_question + sep_token + tokenized_text
            for j in range(no_negs):
                cross_texts.append(cross_text)
                labels.append(0)
                
        neg_ids = [x for x in retrieved_sub_ids if dscorpus['id'][x] not in ans_ids]
        neg_ids = neg_ids[:no_negs]
        for neg_id in neg_ids:
            tokenized_text = dscorpus['tokenized_text'][neg_id]
            cross_text = tokenized_question + sep_token + tokenized_text
            cross_texts.append(cross_text)
            labels.append(1)
                    
    dff = pd.DataFrame()
    dff['cross_text'] = cross_texts
    dff['label'] = labels
        
    return dff

def build_cross_data(dcorpus, tokenizer, json_file, csv_file, no_hard_negs=10, no_negs= 20, top_k= 30):
    """
    This function builds train, val, test dataframe for training and evaluating cross-encoder
    """
    no_random_negs = no_negs - no_hard_negs
    sep_token = " . " + tokenizer.sep_token + " " 
    with open(json_file, 'r') as f:
        retrieved_list = json.load(f)
    df = pd.read_csv(csv_file)    
    cross_texts, labels = [], []
    for i in range(len(df)):
        tokenized_question = df['tokenized_question'][i]
        retrieved_ids = retrieved_list[i][:top_k]
        ans_ids = str(df['ans_id'][i])
        ans_ids = [int(x) for x in ans_ids.split(", ")]
            
        for a_id in ans_ids:
            tokenized_text = dcorpus['tokenized_text'][a_id]
            cross_text = tokenized_question + sep_token + tokenized_text
            for j in range(no_negs):
                cross_texts.append(cross_text)
                labels.append(0)
                
        neg_ids = [x for x in retrieved_ids if x not in ans_ids]
        hard_negs = neg_ids[:no_hard_negs]
        random_negs = neg_ids[no_hard_negs:]
        random_negs = random.sample(random_negs, no_random_negs)
        final_neg_ids = hard_negs + random_negs
        for neg_id in final_neg_ids:
            tokenized_text = dcorpus['tokenized_text'][neg_id]
            cross_text = tokenized_question + sep_token + tokenized_text
            cross_texts.append(cross_text)
            labels.append(1)
                    
    dff = pd.DataFrame()
    dff['cross_text'] = cross_texts
    dff['label'] = labels
        
    return dff

def build_cross_dataloader(df, tokenizer, text_len, batch_size, shuffle = False):
    """
    This function builds train, val, test dataloader for training and evaluating cross-encoder
    """
    cross_texts = df["cross_text"].tolist()
    labels = df["label"]
        
    C = tokenizer.batch_encode_plus(cross_texts, padding='max_length', truncation=True, max_length=text_len, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.long)
    data_tensor = TensorDataset(C['input_ids'], C['attention_mask'], labels)
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader
