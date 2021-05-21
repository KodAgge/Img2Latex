import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
import CorpusHelper
import time
import numpy as np
# import Performance

from CNN import Net as CNN
from paper_LSTM import paper_LSTM_Module
from AttentionMechanism import AttentionMechanism

import sys
sys.path.insert(0, '..\data')
from CROHME_Datasets import CROHME_Training_Set, CROHME_Testing_Set, CROHME_Validation_Set



class EncoderDecoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, batch_size, sequence_length, vocab_size, o_layer_size, v_length=512):
        super().__init__()

        # Static params
        self.v_length = v_length
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.o_layer_size = o_layer_size

        input_size = embedding_size + o_layer_size

        # Network Modules
        self.CNN = CNN()
        self.LSTM_module = paper_LSTM_Module(input_size, hidden_size, batch_size)
        self.AttentionMechanism = AttentionMechanism(beta_size=512, hidden_size=hidden_size, v_length=v_length) # TODO: Change these hard-coded values

        # The other layers
        # self.E = nn.Parameter(torch.zeros(embedding_size, vocab_size)).double()
        self.E = nn.Embedding(vocab_size, embedding_size).double()
        self.O = nn.Linear(v_length + hidden_size, o_layer_size, bias=False).double()  # TODO: ADD BIAS?
        self.W_out = nn.Linear(o_layer_size, vocab_size, bias=False).double()  # TODO: ADD BIAS?
        self.softmax = nn.Softmax(1).double()

        # Initialization of h_t
        self.init_Wh = nn.Linear(v_length, hidden_size).double()


    def init_parameters(self):
        """Function to initialize parameters that are NOT initialized in the modules (which should take care of themselves"""
        pass

    def forward(self, X_batch, labels_batch): 
        # 1) CNN, aka "HyperCube Creation" :) 
        V = self.CNN(X_batch)

        # Transforming into a cube
        V = V.permute(0, 2, 3, 1)
        batch_size, H_prime, W_prime, C = V.shape
        V = torch.reshape(V, (batch_size, H_prime * W_prime, C))

        # Pre-allocate memory
        output = torch.zeros(self.batch_size, self.sequence_length, self.vocab_size).double()

        # Initialize Y and O 
        Y_0 = (self.vocab_size - 3) * torch.ones(self.batch_size).long()
        O_0 = torch.zeros(self.o_layer_size, self.batch_size).double()
        X_t = torch.cat((torch.transpose(self.E(Y_0), 0, 1), O_0), 0)

        # Reset S_t
        self.LSTM_module.reset_LSTM_states()

        # Initialize H_t
        mean_encoder_out = torch.mean(V, 1)
        H_0 = torch.transpose(torch.tanh(self.init_Wh(mean_encoder_out)), 0, 1)
        self.LSTM_module.H_t = H_0

        for i in range(self.sequence_length):
            H_t = self.LSTM_module(X_t)         # 2) LSTM 
            
            # 3) Attention Mechanism
            C_t, _ = self.AttentionMechanism(V, torch.transpose(H_t, 0, 1))  
            
            concat = torch.transpose(torch.cat((H_t, C_t), 0), 0, 1)
            linear_O = self.O(concat) # THIS WAS THE PROBLEM BEFORE
            O_t = torch.tanh(linear_O)
            Q_t = self.W_out(O_t) # This is the wanted output for the cross-entropy, that is un-softmaxed probabilities
            output[:, i, :] = Q_t
            
            # Greedy approach
            # print(torch.argmax(Q_t, dim=1))
            
            O_t = torch.transpose(O_t, 0, 1)
            Y_t = labels_batch[:, i]
            # print(Y_t)


            X_t = torch.cat((torch.transpose(self.E(Y_t), 0, 1), O_t), 0)

        return output


    def forward_predict(self, X_batch): 
        # 1) CNN, aka "HyperCube Creation" :) 
        V = self.CNN(X_batch)

        # Transforming into a cube
        V = V.permute(0, 2, 3, 1)
        batch_size, H_prime, W_prime, C = V.shape
        V = torch.reshape(V, (batch_size, H_prime * W_prime, C))

        # Pre-allocate memory
        output = torch.zeros(self.batch_size, self.sequence_length, self.vocab_size).double()

        # Initialize Y and O 
        Y_0 = (self.vocab_size - 3) * torch.ones(self.batch_size).long()
        O_0 = torch.zeros(self.o_layer_size, self.batch_size).double()

        X_t = torch.cat((torch.transpose(self.E(Y_0), 0, 1), O_0), 0)

        # Reset S_t
        self.LSTM_module.reset_LSTM_states()

        # Initialize H_t
        mean_encoder_out = torch.mean(V, 1)
        H_0 = torch.transpose(torch.tanh(self.init_Wh(mean_encoder_out)), 0, 1)
        self.LSTM_module.H_t = H_0

        for i in range(self.sequence_length):
            H_t = self.LSTM_module(X_t)         # 2) LSTM 

            # 3) Attention Mechanism
            C_t, _ = self.AttentionMechanism(V, torch.transpose(H_t, 0, 1))  

            concat = torch.transpose(torch.cat((H_t, C_t), 0), 0, 1)
            linear_O = self.O(concat) # THIS WAS THE PROBLEM BEFORE
            O_t = torch.tanh(linear_O)
            Q_t = self.W_out(O_t) # This is the wanted output for the cross-entropy, that is un-softmaxed probabilities

            output[:, i, :] = Q_t
            Y_distr = self.softmax(Q_t)
            
            # Greedy approach
            Y_t = torch.argmax(Y_distr, dim=1)
            O_t = torch.transpose(O_t, 0, 1)
            X_t = torch.cat((torch.transpose(self.E(Y_t), 0, 1), O_t), 0)

        return output


    def predict_single(self, loader):
        # Retrieve image
        dataiter = iter(loader)
        data = dataiter.next()
        labels = data["label"]
        images = data["image"].squeeze()
        image = images[0, :, :]
        label = labels[0, :]

        # Make single image ready for forward pass
        test_images = data["image"]
        P = self.forward_predict(test_images)
        P_single = P[0, :, :].squeeze()
        logit = self.softmax(P_single)
        
        # Greedy approach
        predicted_label = torch.argmax(logit, dim=1) + 1

        # Printing
        print("\nTrue label:\n", label)
        print("\nPredicted label:\n", predicted_label)

        print("\nTrue image:")

        # Plot image
        plt.imshow(image)
        plt.show()


    def predict_multi(self, loader):
        # corpusDict = CorpusHelper.corpus()
        
        # Retrieve images
        dataiter = iter(loader)
        data = dataiter.next()
        labels = data["label"]
        images = data["image"].squeeze()
        batch_size = images.shape[0]


        # Make images ready for forward pass
        test_images = data["image"]
        P = self.forward_predict(test_images)

        for i in range(batch_size):
            image = images[i, :, :]
            label = labels[i, :]
            P_single = P[i, :, :].squeeze()
            logit = self.softmax(P_single)
        
            # Greedy approach
            predicted_label = torch.argmax(logit, dim=1) + 1

            # Printing
            print("\nTrue label:\n", label)
            print()
            print(CorpusHelper.unTokenize(label,specialToken = True), "\n")

            print("\nPredicted label:\n", predicted_label)
            print()
            print(CorpusHelper.unTokenize(predicted_label,specialToken = True), "\n")

            print("\nTrue image:")

            # Plot image
            plt.imshow(image)
            plt.show()


    def write_results(self, loader, file_name = "results_test", headers = True):
        print("\nWriting ground truth and predicted labels to " + file_name + ".txt ...")
        results_file = open("project/results/" + file_name + ".txt", "w")

        if headers:
            results_file.write("Ground trutch;Predicted labels\n")

        start_time = time.perf_counter()
        for i, data in enumerate(loader, 0):
            images, labels = data["image"], data["label"] - 1 # Labels måste börja på 0

            logits = self.forward_predict(images)

            predicted_labels = torch.argmax(logits, dim=2) + 1 # greedy

            labels += 1

            for j in range(images.shape[0]):
                results_file.write(str(labels[j, :].tolist()) + ";" + str(predicted_labels[j, :].tolist()) + "\n")

            if (i + 1) % 20 == 0 or i == 0:
                print("\tBatch", i+1, "out of", len(loader), "done.")

        end_time = time.perf_counter()
        results_file.close()
        print("Completed in", end_time - start_time, "seconds!")


def MGD(net, train_dataloader, optimizer, learning_rates, n_epochs, constant_lr = True):
    criterion = nn.CrossEntropyLoss() # Ändra denna?

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        # Changing learning rate according to Stanford paper
        if not constant_lr:
            for g in optimizer.param_groups:
                g['lr'] = learning_rates[epoch]

        print("\n" + "=" * 50 + "\n")
        print("Starting epoch", epoch + 1, "of", n_epochs, "with learning rate ", end="")
        for g in optimizer.param_groups:
            print(g['lr'], "\n")

        for i, data in enumerate(train_dataloader, 0):

            outputs = None
            # get the inputs; data is a list of [images, labels]
            images, labels = data["image"], data["label"] - 1 # Labels måste börja på 0
            
            # forward-pass
            outputs = net(images, labels)

            
            # backwards pass + gradient step
            loss = criterion(outputs.view(-1, 144), labels.view(-1))
            
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward() #retain_graph=True)
            
            optimizer.step()

            if i == 0 and epoch == 0:
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item()

            if i == 0 or (i+1) % 5 == 0:
                print("\tBatch", i+1, "of", len(train_dataloader), "complete")
                print("\t\tLoss =", loss.item())
                print("\t\tSmooth loss =", smooth_loss)

            running_loss += loss.item()

            # batch_num += 1

        print("\nEpoch", epoch + 1, "of", n_epochs, "complete")
        print("Average loss", running_loss / len(train_dataloader))
    
    print("=" * 50 + "\n\nTraining complete!")

    return net


def set_learning_rates(learning_rate_baselines, cut_offs, n_epochs):
    learning_rates = np.zeros([n_epochs])

    for i in range(max(n_epochs, learning_rate_baselines[-1])):
        if i < cut_offs[0] + 1:
            learning_rates[i] = learning_rate_baselines[0]

        if i >= cut_offs[0] + 1 and i < cut_offs[1] - 1:
            learning_rates[i] = learning_rate_baselines[1]
        
        if i >= cut_offs[1] - 1 and i < cut_offs[2]:
            eta_max = - np.log10(learning_rate_baselines[1])
            eta_min = - np.log10(learning_rate_baselines[2])
            eta = eta_min  + (eta_max - eta_min) * (cut_offs[2] - 1 - i) / (cut_offs[2] - cut_offs[1])
            learning_rates[i] = 10 ** -eta

        if i >= cut_offs[2]:
            learning_rates[i] = learning_rate_baselines[2]

    return learning_rates


def main():
    # Load datasets
    train_set = CROHME_Training_Set()
    test_set = CROHME_Testing_Set()
    val_set = CROHME_Validation_Set()
    
    # Hyperparameters
    embedding_size = 80; # number of rows in the E-matrix
    o_layer_size = 100;  # size of o-vektorn TODO: What should this be?
    hidden_size = 512; 
    sequence_length = 109; vocab_size = 144; 
    batch_size = 2
    n_epochs = 20

    # Learning rates
    constant_lr = False # False to use changing learning rate suggest by stanford
    learning_rate_levels = [1e-4, 1e-3, 1e-5]
    cutoffs_learning_rates = [1, 9, 15]
    learning_rates = set_learning_rates(learning_rate_levels, cutoffs_learning_rates, n_epochs)

    # Putting datasets in DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Loading existing model?
    load_model = False
    load_path = "project/saved_models/"
    load_name_prefix = "SAVED"

    # Saving model after training?
    save_model = True
    save_name_prefix = "TR2000NE2"

    # Writing predictions to a .txt file?
    write_results_train = True
    write_results_test = True
    results_name_suffix = "TR2000NE2"

    # Print a few predicted examples?
    print_examples_test = True
    print_examples_train = True

    # Initializing model
    ED = EncoderDecoder(embedding_size=embedding_size, hidden_size=hidden_size, batch_size=batch_size, sequence_length=sequence_length, vocab_size=vocab_size, o_layer_size = o_layer_size)
    optimizer = optim.Adam(ED.parameters(), lr=learning_rates[0])

    # Loading the model
    if load_model:
        print("\nLoading:")
        print("\t" + load_path + load_name_prefix + "_MODEL")
        print("\t" + load_path + load_name_prefix + "_OPTIMIZER")
        ED.load_state_dict(torch.load(load_path + load_name_prefix + "_MODEL"))
        optimizer.load_state_dict(torch.load(load_path + load_name_prefix + "_OPTIMIZER"))

    # Training the model
    ED_Trained = MGD(ED, train_loader, optimizer, learning_rates, n_epochs, constant_lr)

    # Saving the model
    if save_model:
        print("\nSaving:")
        print("\t" + load_path + save_name_prefix + "_MODEL")
        print("\t" + load_path + save_name_prefix + "_OPTIMIZER")
        torch.save(ED_Trained.state_dict(), load_path + save_name_prefix + "_MODEL")
        torch.save(optimizer.state_dict(), load_path + save_name_prefix + "_OPTIMIZER")

    # Writing results
    if write_results_train:
        ED_Trained.write_results(train_loader, "TRAIN_" + results_name_suffix)
    
    if write_results_test:
        ED_Trained.write_results(test_loader, "TEST_" + results_name_suffix)

    # Printing examples
    if print_examples_test:
        print("\n\nEXAMPLES FROM THE TEST SET:")
        ED_Trained.predict_multi(test_loader)

    if print_examples_train:
        print("\n\nEXAMPLES FROM THE TRAIN SET:")
        ED_Trained.predict_multi(train_loader)



if __name__=='__main__':
    main()
    
