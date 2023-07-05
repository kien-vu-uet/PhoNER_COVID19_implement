import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer

class COVID19Dataset(Dataset):
    def __init__(self, **kwargs):
        self.raw_data = json.load(open(kwargs['data_path'], 'r'))
        self.map_data = {}
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['tokenizer'])
        self.max_length = kwargs['max_length']
        self.tag2idx = {
            'O': 0,
            'B-ORGANIZATION': 1,
            'I-ORGANIZATION': 2,
            'B-SYMPTOM_AND_DISEASE': 3,
            'I-SYMPTOM_AND_DISEASE': 4,
            'B-LOCATION': 5,
            'I-LOCATION': 6,
            'B-PATIENT_ID': 7,
            'I-PATIENT_ID': 8,
            'B-DATE': 9,
            'I-DATE': 10,
            'B-AGE': 11,
            'I-AGE': 12,
            'B-NAME': 13,
            'I-NAME': 14,
            'B-JOB': 15,
            'I-JOB': 16,
            'B-TRANSPORTATION': 17,
            'I-TRANSPORTATION': 18,
            'B-GENDER': 19,
            'I-GENDER': 20,
            '<PAD>': 21
        }
    
    @property
    def tag_vocab_size(self):
        return len(self.tag2idx)
    
    @property
    def pad_tag_id(self):
        return self.tag2idx['<PAD>']

    @property
    def outline_tag_id(self):
        return self.tag2idx['O']

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        def tokenize(tokenizer, words, max_length):
            results = tokenizer.encode(words, add_special_tokens=False)
            results.insert(0, tokenizer.cls_token_id)
            results.append(tokenizer.sep_token_id)
            attn_mask = [1.] * len(results)
            while len(results) < max_length: results.append(tokenizer.pad_token_id)
            while len(attn_mask) < max_length: attn_mask.append(0.)
            return results[:max_length], attn_mask[:max_length]

        if index not in self.map_data.keys():
            item = self.raw_data[index]
            input_ids, attn_mask = tokenize(self.tokenizer, item['words'], self.max_length)
            target_tags = [self.tag2idx[tag] for tag in item['tags']]
            target_tags.insert(0, self.pad_tag_id)
            target_tags.append(self.pad_tag_id)
            while len(target_tags) < self.max_length: target_tags.append(self.pad_tag_id)
            self.map_data[index] = (
                torch.tensor(input_ids), 
                torch.tensor(attn_mask), 
                torch.tensor(target_tags[:self.max_length])
                )

        return self.map_data[index]
    

if __name__ == "__main__": 
    kwargs = {
        'data_path' : '../data/syllable/train_syllable.json',
        'tokenizer' : 'vinai/phobert-base-v2',
        'max_length' : 100, 
        'pad_tag' : 0
    }

    dataset = COVID19Dataset(**kwargs)
    print([i.size() for i in dataset.__getitem__(1)])
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, )
    for i, (input_ids, attn_mask, target_tags) in enumerate(loader):
        
        print(input_ids.size(), attn_mask.size(), target_tags.size())
        break
        