import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class paper_LSTM_Module(nn.Module):  # TODO: multiple LSTM:s on top of each other (num_layers)?, directions?

  def __init__(self, input_size, hidden_size, batch_size):
    """ Sets the size of the input at each timestep (input_size) and the size 
    of the hidden vector (hidden_size) and initializes weights. Note that the size of the cell state
    is the same hidden_size per definition. """

    super().__init__()
    self.input_size = input_size  # Size of the input-values in sequence
    self.hidden_size = hidden_size  # Note that the size of the cellstate is equal to the hidden_size in the standard architecture
    self.batch_size = batch_size

    self.S_t = None  # cell state!
    self.H_t = None  # hidden state!

    # Forget-gate layer parameters
    self.Wf = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size, dtype=torch.double))
    self.bf = nn.Parameter(torch.zeros(hidden_size, 1, dtype=torch.double))

    # Input-gate layer parameters
    self.Wi = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size, dtype=torch.double))
    self.bi = nn.Parameter(torch.zeros(hidden_size, 1, dtype=torch.double))

    # Candidate parameters
    self.Wc = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size, dtype=torch.double))
    self.bc = nn.Parameter(torch.zeros(hidden_size, 1, dtype=torch.double))

    # Output-gate layer parameters
    self.Wo = nn.Parameter(torch.zeros(hidden_size, hidden_size + input_size, dtype=torch.double))
    self.bo = nn.Parameter(torch.zeros(hidden_size, 1, dtype=torch.double))

    self.init_weights()
    print(self.Wc)
    print(torch.sum(self.Wc))
    print(self.bc)
    print(torch.sum(self.bc))
    input('initialiseringen')

    self.reset_LSTM_states()

  def reset_LSTM_states(self):
    """Since these are not model parameters (we don't want to find their gradients) we initialize them for each timestep to be:"""
    self.H_t = torch.zeros(self.hidden_size, self.batch_size, dtype=torch.double)
    self.S_t = torch.zeros(self.hidden_size, self.batch_size, dtype=torch.double)


  def init_weights(self):   # TODO: CHANGE THIS? It initializes the bias-terms as well now
    """ Sets the weights in a standard way. """

    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
        weight.data.uniform_(-stdv, stdv)

  def forward(self, X_t): 
    # OBS!!! since x_t = [E*y(t-1), o(t-1)]
    # E -> [n1, |V|], o(t-1)-> [n2, 1]. Thus:
    # X_t -> [n1+n2, batch_size]
    # X_and_H -> [n1+n2+hidden_size] = 
    X_and_H = torch.cat((X_t, self.H_t), 0)
    
    # TODO: Make sure the dimensionality is: [input_size, batch_size]

    # Update cell state (S_t)
    F_t = torch.sigmoid(self.Wf @ X_and_H + self.bf).double()
    I_t = torch.sigmoid(self.Wi @ X_and_H + self.bi).double()
    C_t = torch.tanh(self.Wc @ X_and_H + self.bc) .double() 
    self.S_t = F_t * self.S_t + I_t * C_t  # Hadamard product

    # Update hidden state (H_t)
    O_t = torch.sigmoid(self.Wo @ X_and_H + self.bo).double()
    self.H_t = O_t * torch.tanh(self.S_t)

    return self.H_t
