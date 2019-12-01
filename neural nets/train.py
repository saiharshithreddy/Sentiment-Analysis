from yelp_data import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
import pandas as pd


if __name__=='__main__':
    config = Config()
    train_file = pd.read_csv('../data/yelp_train.csv')
    test_file = pd.read_csv('../data/yelp_test.csv')
    # for testing 
    #train_file = train_file.iloc[:1000,:]
    #test_file = test_file.iloc[:1000,:]
    
    print("data reading done")
    # Glove embeddings
    w2v_file = '../data/glove.840B.300d.txt'
    
    dataset = Yelp_Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file)
    print("data loaded")
    
    # call the model
    model = RCNN(config, len(dataset.vocab), dataset.word_embeddings)
    # if gpu
    if torch.cuda.is_available():
        model.cuda()
    # train 
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    
    
    train_losses = []
    val_accuracies = []
    
    # epochs 15
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = get_accuracy(model, dataset.train_iterator)
    val_acc = get_accuracy(model, dataset.val_iterator)
    test_acc = get_accuracy(model, dataset.test_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))