train_dataset_config = {
        'data_path' : '../data/syllable/train_syllable.json',
        'tokenizer' : 'vinai/phobert-base-v2',
        'max_length' : 100, 
    }


val_dataset_config = train_dataset_config.copy()
val_dataset_config['data_path'] = '../data/syllable/dev_syllable.json'