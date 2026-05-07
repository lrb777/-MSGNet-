import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MSGNet-main'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'MSGNet-main'))

import argparse, torch
from data_provider.data_factory import data_provider

args = argparse.Namespace(
    data='custom',
    root_path=os.path.join('..', 'data', 'processed'),
    data_path='etf_returns.csv',
    features='M', target='医药', freq='b',
    seasonal_patterns='Monthly',
    seq_len=60, label_len=30, pred_len=5,
    embed='timeF', num_workers=0, batch_size=32,
)

train_set, train_loader = data_provider(args, 'train')
val_set,   _            = data_provider(args, 'val')
test_set,  _            = data_provider(args, 'test')

print(f'train samples : {len(train_set)}')
print(f'val   samples : {len(val_set)}')
print(f'test  samples : {len(test_set)}')

x, y, xm, ym = next(iter(train_loader))
print(f'batch_x shape : {tuple(x.shape)}   # [B, seq_len, N]')
print(f'batch_y shape : {tuple(y.shape)}   # [B, label+pred, N]')
print(f'GPU available : {torch.cuda.is_available()}')
