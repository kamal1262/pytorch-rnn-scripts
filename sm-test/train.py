# Based on github.com/pytorch/examples/blob/master/word_language_model
import argparse
import math
import os
from shutil import copy
import time
import torch
import torch.nn as nn



# kamal will print now

print("========================================================================")
print(" Did we managed to import scipy module, just checking !!!")
from scipy.spatial import distance
print("========================================================================")
print(" if successful, it should not have any error!")


import data
from rnn import RNNModel

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')

# Hyperparameters sent by the client are passed as command-line arguments to the script.
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', type=bool, default=False,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

# Data and model checkpoints/otput directories from the container environment
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

args = parser.parse_args()

print(args)

model_path = os.path.join(args.model_dir, 'model.pth')
model_info_path = os.path.join(args.model_dir, 'model_info.pth')

checkpoint_path = os.path.join(args.output_data_dir, 'model.pth')
checkpoint_state_path = os.path.join(args.output_data_dir, 'model_info.pth')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Load data
###############################################################################

print('Load data')
corpus = data.Corpus(args.data_dir)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ??? a g m s ???
# ??? b h n t ???
# ??? c i o u ???
# ??? d j p v ???
# ??? e k q w ???
# ??? f l r x ???.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


print('Batchify dataset')
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

print('Build the model')
ntokens = len(corpus.dictionary)
rnn_type = 'LSTM'
model = RNNModel(rnn_type, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

# Save the data into model dir to be used with the model later
for file_name in os.listdir(args.data_dir):
    full_file_name = os.path.join(args.data_dir, file_name)
    if os.path.isfile(full_file_name):
        copy(full_file_name, args.model_dir)

# Save arguments used to create model for restoring the model later
with open(model_info_path, 'wb') as f:
    model_info = {
        'rnn_type': rnn_type,
        'ntoken': ntokens,
        'ninp': args.emsize,
        'nhid': args.nhid,
        'nlayers': args.nlayers,
        'dropout': args.dropout,
        'tie_weights': args.tied
    }
    torch.save(model_info, f)


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ??? a g m s ??? ??? b h n t ???
# ??? b h n t ??? ??? c i o u ???
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // args.bptt, lr,
                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_state = None

print('Starting training.')
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_state or val_loss < best_state['val_loss']:
        best_state = {
            'epoch': epoch,
            'lr': lr,
            'val_loss': val_loss,
            'val_ppl': math.exp(val_loss),
        }
        print('Saving the best model: {}'.format(best_state))
        with open(checkpoint_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(checkpoint_state_path, 'w') as f:
            f.write('epoch {:3d} | lr: {:5.2f} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, lr, val_loss, math.exp(val_loss)))
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0

# Load the best saved model.
with open(checkpoint_path, 'rb') as f:
    model.load_state_dict(torch.load(f))
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# Move the best model to cpu and resave it
with open(model_path, 'wb') as f:
    torch.save(model.cpu().state_dict(), f)
