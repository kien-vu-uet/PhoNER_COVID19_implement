from transformers import AutoModel
import torch
import torch.nn as nn
import os
import time
from dataset import COVID19Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PhoBertNER(nn.Module):

    def __init__(self, **kwargs):
        super(PhoBertNER, self).__init__()
        self.phobert = AutoModel.from_pretrained(kwargs['model'])
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.classifier = nn.Linear(self.phobert.config.hidden_size, kwargs['num_cls'])

    def forward(self, input_ids, attn_mask, apply_softmax=False):
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        outputs = self.phobert(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        outputs = self.dropout(outputs.last_hidden_state)
        outputs = self.classifier(outputs)
        if apply_softmax: return torch.softmax(outputs, dim=-1)
        return outputs
    
def move_to_cuda(batch, device):
    return (t.to(device) for t in batch)

def update_confusion_matrix(cm, input, target, ignore_index, outline_index=0):
    assert input.size() == target.size()
    input[input == ignore_index] = outline_index
    input = input[target != ignore_index].view(-1).tolist()
    target = target[target != ignore_index].view(-1).tolist()
    for i in range(len(target)):
        cm[target[i], input[i]] += 1
    return cm

def plot_confusion_matrix(cm, class_names, fig_path='foo.png'):
    with plt.rc_context({'figure.facecolor':'white'}):
        plt.figure(figsize=[10, 8])

        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="crest")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=10)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=10)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(fig_path)
        plt.close(fig_path)

def compute_score(cm, metric='accuracy', eps=1e-12):
    tp = torch.diag(cm)
    fp = torch.sum(cm, axis=0) - tp
    fn = torch.sum(cm, axis=1) - tp
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall)
    if metric == 'f1_macro': return torch.sum(f1) * 1.0 / f1.size(0)
    elif metric == 'f1_weighted': return torch.sum(f1 * torch.sum(cm, axis=1)) * 1.0 / torch.sum(cm)
    else: return torch.sum(tp) * 1.0 / torch.sum(cm)


def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_path):
    tag_vocab_size = train_loader.dataset.tag_vocab_size
    pad_tag_id = train_loader.dataset.pad_tag_id
    outline_tag_id = train_loader.dataset.outline_tag_id
    cls_names = list(train_loader.dataset.tag2idx.keys())

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    best_model_params_path = os.path.join(ROOT_DIR, model_path)
    best_val_loss = 100.
    for ep in range(1, epochs + 1):
        avg_train_loss = 0.
        train_cm = torch.zeros((tag_vocab_size-1, tag_vocab_size-1), dtype=torch.long)

        for i, batch in enumerate(train_loader):
            input_ids, attn_mask, target_tags = move_to_cuda(batch, device)
            start = time.time()
            optimizer.zero_grad()
            logit_tags = model(input_ids, attn_mask)
            loss = criterion(logit_tags.view(-1, logit_tags.size(-1)), target_tags.view(-1))
            loss.backward()
            optimizer.step()

            train_cm = update_confusion_matrix(train_cm, logit_tags.argmax(dim=-1), target_tags, pad_tag_id, outline_tag_id)
            
            avg_train_loss = (avg_train_loss * i + loss) / (i + 1)
            s_per_batch = time.time() - start
            print(f'| epoch {ep:3d} | | {(i+1):5d}/{len(train_loader):5d} batches | '
                f'| s/batch {s_per_batch:5.2f} | '
                f'| loss {int(loss):4d}.{int((loss - int(loss))*100):02d} | '
                f'| avg_loss {int(avg_train_loss):4d}.{int((avg_train_loss - int(avg_train_loss))*100):02d} |')
        # For evalution
        print('|-------------------------------------------------------------------------------------------|')

        print(f"\nTrain: f1_macro     {compute_score(train_cm, metric='f1_macro'):0.2f}\n"
              f"       f1_weighted  {compute_score(train_cm, metric='f1_weighted'):0.2f}\n"
              f"       accuracy     {compute_score(train_cm, metric='accuracy'):0.2f}\n")
        plot_confusion_matrix(train_cm, cls_names[:-1], fig_path=f'heatmap/train/ep_{ep}.png')

        val_loss = 0.
        start = time.time()        
        val_cm = torch.zeros_like(train_cm, dtype=torch.long)

        for i, batch in enumerate(val_loader):
            input_ids, attn_mask, target_tags = move_to_cuda(batch, device)
            start = time.time()
            with torch.no_grad(): 
                logit_tags = model(input_ids, attn_mask)
                loss = criterion(logit_tags.view(-1, logit_tags.size(-1)), target_tags.view(-1))
                val_loss += loss
                val_cm = update_confusion_matrix(val_cm, logit_tags.argmax(dim=-1), target_tags, pad_tag_id, outline_tag_id)
               
        val_loss /= len(val_loader)
        
        print(f"Val:   loss         {val_loss:0.4f}\n"
              f"       f1_macro     {compute_score(val_cm, metric='f1_macro'):0.2f}\n"
              f"       f1_weighted  {compute_score(val_cm, metric='f1_weighted'):0.2f}\n"
              f"       accuracy     {compute_score(val_cm, metric='accuracy'):0.2f}\n")
        plot_confusion_matrix(val_cm, cls_names[:-1], fig_path=f'heatmap/val/ep_{ep}.png')


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
            print('Save model!\n')

        print('|-------------------------------------------------------------------------------------------|')


if __name__ == '__main__':
    BATCH_SIZE = 16
    EPOCHS = 25
    LR = 3e-5
    BEST_MODEL_PARAMS_PATH = "best_model_params_syllable_large.pt"

    # Prepare data
    train_dataset_config = {
        'data_path' : 'data/syllable/train_syllable.json',
        'tokenizer' : 'vinai/phobert-large',
        'max_length' : 100, 
    }
    train_set = COVID19Dataset(**train_dataset_config)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset_config = {
        'data_path' : 'data/syllable/dev_syllable.json',
        'tokenizer' : train_dataset_config['tokenizer'],
        'max_length' : train_dataset_config['max_length'], 
    }
    val_set = COVID19Dataset(**val_dataset_config)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Define model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhoBertNER(model = train_dataset_config['tokenizer'], num_cls = train_set.tag_vocab_size, dropout=0.1)
    # model.load_state_dict(torch.load(BEST_MODEL_PARAMS_PATH))
    model = model.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=train_set.pad_tag_id)

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=LR)

    train(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device, BEST_MODEL_PARAMS_PATH)
    
    