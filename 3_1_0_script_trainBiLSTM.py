from trainer import train
from models import BiLSTM
args = {
        'seed': 123,
        'train_json': 'train_fbank.json',
        'val_json': 'dev_fbank.json',
        'test_json': 'test_fbank.json',
        'batch_size': 4,
        'num_layers': 1,
        'fbank_dims': 23,
        'model_dims': 128,
        'concat': 1,
        'lr': 0.5,
        'vocab': 'vocab_39.txt',
        'report_interval': 50,
        'num_epochs': 15,
}
# Instantiate the model
model = BiLSTM(args['num_layers'], args['fbank_dims'], args['model_dims'], len(args['vocab']))

# Start training
model_path = train(model, args)