import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class paperLSTM(nn.Module):  # TODO: multiple LSTM:s on top of each other (num_layers)?, directions?
  """ Own implementation of LSTM.
  A single layer LSTM progressing in one single direction. The implemented equations for 
  updating cell and hidden state can be found at https://colah.github.io/posts/2015-08-Understanding-LSTMs/. """

  def __init__(self, input_size, hidden_size):
    """ Sets the size of the input at each timestep (input_size) and the size 
    of the hidden vector (hidden_size) and initializes weights. Note that the size of the cell state
    is the same hidden_size per definition. """

    super().__init__()
    self.input_size = input_size  # Size of the input-values in sequence
    self.hidden_size = hidden_size  # Note that the size of the cellstate is equal to the hidden_size in the standard architecture

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
    print('hjoijijijijijij')

  def init_weights(self):   # TODO: CHANGE THIS?
    """ Sets the weights in a standard way. """

    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
        weight.data.uniform_(-stdv, stdv)