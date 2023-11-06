import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer
from util import build_cross_dataloader
from trainer import CrossTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorpus_file", default=None, type=str,
                        help="Directory of the sub corpus file (scorpus.json)")
    parser.add_argument("--file_dir", default=None, type=str,
                        help="Directory of the files to prepare data for training cross-encoder")
    parser.add_argument("--json_dir", default=None, type=str,
                        help="Directory of the files to prepare data for training cross-encoder")
    parser.add_argument("--cross_checkpoint", default="vinai/phobert-base-v2", type=str,
                        help="Name or directory of pretrained model for cross-encoder")
    parser.add_argument("--cross_representation", default=0, type=int,
                        help="Type of encoder representation (-10 for avg, -100 for pooled-output)")
    parser.add_argument("--cross_num_epochs", default=2, type=int,
                        help="Number of training epochs for cross-encoder")
    parser.add_argument("--cross_max_len", default=256, type=int,
                        help="Maximum token length for cross-encoder input")
    parser.add_argument("--cross_batch_size", default=1, type=int,
                        help="Cross-encoder training batch size (sum in all gpus)")
    parser.add_argument("--cross_lr", default=0.00001, type=float,
                        help="cross-encoder training learning rate")
    parser.add_argument("--cross_dropout", default=0.1, type=float,
                        help="cross-encoder threshold for classification")
    parser.add_argument("--cross_no_negs", default=1, type=int,
                        help="Number of top negatives using")
    parser.add_argument("--cross_eval_steps", default=1000, type=int)
    parser.add_argument("--cross_patience", default=10, type=int)
    parser.add_argument("--cross_best_path", default=None, type=str,
                        help="Path to save the best state")
    parser.add_argument("--cross_final_path", default=None, type=str,
                        help="Path to save the final state")
    parser.add_argument("--cross_load_path", default=None, type=str,
                        help="Path to load state")
    
    args = parser.parse_args()
    
    dscorpus = pd.read_csv(args.scorpus_file)
    tokenizer = AutoTokenizer.from_pretrained(args.cross_checkpoint)
    train_dataloader = build_cross_dataloader(dscorpus=dscorpus,
                                              json_file=os.path.join(args.json_dir, "dpr_train_sub_retrieved.json"),
                                              csv_file=os.path.join(args.file_dir, "ttrain.csv"),
                                              tokenizer=tokenizer,
                                              text_len=args.cross_max_len,
                                              batch_size=args.cross_batch_size,
                                              no_negs=args.cross_no_negs,
                                              shuffle=True)
    val_dataloader = build_cross_dataloader(dscorpus=dscorpus,
                                            json_file=os.path.join(args.json_dir, "dpr_val_sub_retrieved.json"),
                                            csv_file=os.path.join(args.file_dir, "tval.csv"),
                                            tokenizer=tokenizer,
                                            text_len=args.cross_max_len,
                                            batch_size=args.cross_batch_size,
                                            no_negs=args.cross_no_negs,
                                            shuffle=False)  
    test_dataloader = build_cross_dataloader(dscorpus=dscorpus,
                                            json_file=os.path.join(args.json_dir, "dpr_test_sub_retrieved.json"),
                                            csv_file=os.path.join(args.file_dir, "ttest.csv"),
                                            tokenizer=tokenizer,
                                            text_len=args.cross_max_len,
                                            batch_size=args.cross_batch_size,
                                            no_negs=args.cross_no_negs,
                                            shuffle=False)
    
    trainer = CrossTrainer(args=args,
                           train_loader=train_dataloader,
                           val_loader=val_dataloader,
                           test_loader=test_dataloader)
    # Train the model
    trainer.train_crossencoder()

    torch.cuda.empty_cache()   
if __name__ == "__main__":
    main()