from phobert_fine_tuning import *
from dataset import COVID19Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def infer(model, loader, criterion, device):
    tag_vocab_size = loader.dataset.tag_vocab_size
    pad_tag_id = loader.dataset.pad_tag_id
    outline_tag_id = loader.dataset.outline_tag_id
    cls_names = list(loader.dataset.tag2idx.keys())

    # For evalution
    print('|-------------------------------------------------------------------------------------------|')

    test_loss = 0.      
    cm = torch.zeros((tag_vocab_size-1, tag_vocab_size-1), dtype=torch.long)

    for i, batch in enumerate(loader):
        input_ids, attn_mask, target_tags = move_to_cuda(batch, device)
        with torch.no_grad(): 
            logit_tags = model(input_ids, attn_mask)
            loss = criterion(logit_tags.view(-1, logit_tags.size(-1)), target_tags.view(-1))
            test_loss += loss
            cm = update_confusion_matrix(cm, logit_tags.argmax(dim=-1), target_tags, pad_tag_id, outline_tag_id)
            
    test_loss /= len(loader)
    
    print(f"Val:   loss         {test_loss:0.4f}\n"
            f"       f1_macro     {compute_score(cm, metric='f1_macro'):0.2f}\n"
            f"       f1_weighted  {compute_score(cm, metric='f1_weighted'):0.2f}\n"
            f"       accuracy     {compute_score(cm, metric='accuracy'):0.2f}\n")
    plot_confusion_matrix(cm, cls_names[:-1], fig_path=f'../heatmap/test/test.png')

    print('|-------------------------------------------------------------------------------------------|')

if __name__ == "__main__":
    BATCH_SIZE = 64
    BEST_MODEL_PARAMS_PATH = "best_model_params_word_large.pt"

    # Prepare data
    test_dataset_config = {
        'data_path' : '../data/word/test_word.json',
        'tokenizer' : 'vinai/phobert-large',
        'max_length' : 100, 
    }
    test_set = COVID19Dataset(**test_dataset_config)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Define model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhoBertNER(model = test_dataset_config['tokenizer'], num_cls = test_set.tag_vocab_size, dropout=0.1)
    model.load_state_dict(torch.load(BEST_MODEL_PARAMS_PATH))
    model = model.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=test_set.pad_tag_id)

    infer(model, test_loader, criterion, device)