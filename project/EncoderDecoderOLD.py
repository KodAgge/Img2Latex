import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
import CorpusHelper
import time
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

        # Initialize Y and O 
        output = torch.zeros(self.batch_size, self.sequence_length, self.vocab_size).double()

        Y_0 = (self.vocab_size - 3) * torch.ones(self.batch_size).long()

        O_0 = torch.zeros(self.o_layer_size, self.batch_size).double()

        X_t = torch.cat((torch.transpose(self.E(Y_0), 0, 1), O_0), 0)

        self.LSTM_module.reset_LSTM_states()  # COMMENT IN LINE 46 IN LSTM TO MAKE THIS VERSION WORK!

        # Initialize H_t
        # mean_encoder_out = torch.mean(V, 1)
        # H_0 = torch.tanh(self.init_Wh(mean_encoder_out))
        # self.LSTM_module.H_t = H_0

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

        # Initialize Y and O 
        output = torch.zeros(self.batch_size, self.sequence_length, self.vocab_size).double()

        Y_0 = (self.vocab_size - 3) * torch.ones(self.batch_size).long()
        O_0 = torch.zeros(self.o_layer_size, self.batch_size).double()
        X_t = torch.cat((torch.transpose(self.E(Y_0), 0, 1), O_0), 0)

        self.LSTM_module.reset_LSTM_states()  # THIS WAS THE PROBLEM BEFORE

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

            print("\tBatch", i+1, "out of", len(loader), "done.")

        end_time = time.perf_counter()
        results_file.close()
        print("Completed in", end_time - start_time, "seconds!")


def MGD(net, train_dataloader, learning_rate, n_epochs):
    criterion = nn.CrossEntropyLoss() # Ändra denna?

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        # Changing learning rate according to Stanford paper
        first_cutoff = 1
        second_cutoff = 2
        third_cutoff = 4
        if epoch == first_cutoff:
            for g in optimizer.param_groups:
                g['lr'] = 1e-3
        elif epoch >= second_cutoff and epoch < third_cutoff:
            for g in optimizer.param_groups:
                exponent = 5 + (3 - 5) * (third_cutoff - epoch) / (third_cutoff - second_cutoff)
                g['lr'] = 10 ** -exponent

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
            print("\tBatch", i+1, "of", len(train_dataloader), "complete")
            print("\t\tLoss =", loss.item())

            running_loss += loss.item()

            # batch_num += 1

        print("\nEpoch", epoch + 1, "of", n_epochs, "complete")
        print("Average loss", running_loss / len(train_dataloader))
    
    print("=" * 50 + "\n\nTraining complete!")

    # KOD SOM PREDICTAR PÅ TEST...


    # KOD FÖR ATT KOLLA PÅ NÄTETS PERFORMANCE
    # predictions = []
    # ground_truth = []
    # performance = Performance(predictions, ground_truth)
    # lev_labels, lev_scores = performance.levenshtein()
    # bleu_labels, bleu_scores = performance.bleu()
    # _, latex_scores = performance.equal_latex(expressions1, expressions2)

    # performance.get_statistics(lev_scores, bleu_scores)
    # performance.get_performance(lev_labels, lev_scores, bleu_labels, bleu_scores)

    return net



def main():
    train_set = CROHME_Training_Set()
    test_set = CROHME_Testing_Set()
    val_set = CROHME_Validation_Set()
    
    embedding_size = 80; # number of rows in the E-matrix
    o_layer_size = 100;  # size of o-vektorn TODO: What should this be?
    hidden_size = 512; 
    sequence_length = 109; vocab_size = 144; 

    batch_size = 2

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    

    ED = EncoderDecoder(embedding_size=embedding_size, hidden_size=hidden_size, batch_size=batch_size, sequence_length=sequence_length, vocab_size=vocab_size, o_layer_size = o_layer_size)
    
    ED_Trained = MGD(ED, train_loader, learning_rate=5e-4, n_epochs=1)

    file_name = "TEST"
    ED_Trained.write_results(test_loader, "TEST_" + file_name)
    # ED_Trained.write_results(train_loader, "TRAIN_" + file_name)

    # ED_Trained.predict_multi(train_loader)
    # ED_Trained.predict_multi(test_loader)



if __name__=='__main__':
    main()
    
