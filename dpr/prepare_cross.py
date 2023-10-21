import os
import math
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from util import build_cross_data
from retriever import DPRRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default=None, type=str,
                        help="Directory of the corpus file (corpus.json)")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Directory of the data folder (containing train, test, val, ttrain, ttest, tval file)")
    parser.add_argument("--BE_checkpoint", default="vinai/phobert-base-v2", type=str,
                        help="Name or directory of pretrained model for encoder")
    parser.add_argument("--BE_representation", default=0, type=int,
                        help="Type of encoder representation (-10 for avg, -100 for pooled-output)")
    parser.add_argument("--BE_score", default="dot", type=str,
                        help="Type of similarity score")
    parser.add_argument("--q_fixed", default=False, type=bool,
                        help="To fix question encoder during training stage or not")
    parser.add_argument("--ctx_fixed", default=False, type=bool,
                        help="To fix context encoder during training stage or not")
    parser.add_argument("--q_len", default=32, type=int,
                        help="Maximum token length for question")
    parser.add_argument("--ctx_len", default=256, type=int,
                        help="Maximum token length for context")
    parser.add_argument("--biencoder_path", default=None, type=str,
                        help="Path to save the state with highest validation result.")
    parser.add_argument("--index_path", default=None, type=str,
                        help="Path to save the index")
    
    args = parser.parse_args()
    dpr_retriever = DPRRetriever(args, sub=True)
    dpr_retriever.test_on_data(top_k = [1,5,10,30,100])
    
    
if __name__ == "__main__":
    main()
    