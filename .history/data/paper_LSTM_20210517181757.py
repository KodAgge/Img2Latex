import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class paper_LSTM_Module(nn.Module):  # TODO: multiple LSTM:s on top of each other (num_layers)?, directions?

  def __init__(self, input_size, hidden_size):
    """ Sets the size of the input at each timestep (input_size) and the size 
    of the hidden vector (hidden_size) and initializes weights. Note that the size of the cell state
    is the same hidden_size per definition. """

    super().__init__()
    self.input_size = input_size  # Size of the input-values in sequence
    self.hidden_size = hidden_size  # Note that the size of the cellstate is equal to the hidden_size in the standard architecture
    
    self.S_t = None  # cell state!
    self.H_t = None  # hidden state!

    # Forget-gate layer parameters
    self.Wf = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size)) # dims på denna
    self.bf = nn.Parameter(torch.zeros(hidden_size, 1))

    # Input-gate layer parameters
    self.Wi = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size))
    self.bi = nn.Parameter(torch.zeros(hidden_size, 1))

    # Candidate parameters
    self.Wc = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size))
    self.bc = nn.Parameter(torch.zeros(hidden_size, 1))

    # Output-gate layer parameters
    self.Wo = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size))
    self.bo = nn.Parameter(torch.zeros(hidden_size, 1))

    self.init_weights()

  def init_weights(self):   # TODO: CHANGE THIS?
    """ Sets the weights in a standard way. """

    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
        weight.data.uniform_(-stdv, stdv)

def forward(self, X, , S_t = torch.zeros(self.hidden_size, batch_size)): 
    # TODO: Ändra parametrarna ovan!

    # The forward pass for each sequence in batch
    hidden_sequence = []
    for t in range(sequence_length):
      X_t = X[t, :, :]  # extracts the input vector at timestep t for each sequence in the batch, dim: [input_size, batch_size]. For an image the t:th row is the input vector.
      X_and_H = torch.cat((X_t, H_t), 0)

      # Update cell state (S_t)
      F_t = torch.sigmoid(self.Wf @ X_and_H + self.bf)
      I_t = torch.sigmoid(self.Wi @ X_and_H + self.bi)
      C_t = torch.tanh(self.Wc @ X_and_H + self.bc)  
      S_t = F_t * S_t + I_t * C_t  # Hadamard product

      # Update hidden state (H_t)
      O_t = torch.sigmoid(self.Wo @ X_and_H + self.bo)
      H_t = O_t * torch.tanh(S_t)

      hidden_sequence.append(H_t.unsqueeze(0)) # unsqueeze necessary for concatenation at the end

    # Concatenate and reshape
    hidden_sequences = torch.cat(hidden_sequence)
    hidden_sequences = torch.transpose(hidden_sequences, 1, 2)
    hidden_sequences = torch.transpose(hidden_sequences, 0, 1)

    # A regular LSTM would have the return statement below...    
    return hidden_sequences, (H_t, S_t) # hidden_sequences format: [seq_length, batch_size, hidden_size]
