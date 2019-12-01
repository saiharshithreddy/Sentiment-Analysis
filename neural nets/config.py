class Config(object):
    embed_size = 300 # Size of the word embeddings
    hidden_layers = 1
    hidden_size = 64
    output_size = 2 # no of output labels
    max_epochs = 15
    hidden_size_linear = 64 
    lr = 0.5 # learning rate
    batch_size = 128
    seq_len = None # Sequence length for RNN
    dropout_keep = 0.8
