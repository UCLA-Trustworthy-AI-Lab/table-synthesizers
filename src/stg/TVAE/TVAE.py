"""CLI."""
import numpy as np
import torch
from torch import optim
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import warnings
import time
import threading

from ..base import BaseSynthesizer


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


class TransformerInfo:
    def __init__(self, is_categorical, output_width):
        self.is_categorical = is_categorical
        self.output_width = output_width
        self.is_continuous = not is_categorical

class TableTransformerInfo:
    def __init__(self, transform_info):
        self.transformers = []
        self.output_width = 0
        for col_name, elements in transform_info.items():
            is_categorical = elements['original_dtype'] == 'categorical'
            st, ed = elements['start_idx'], elements['end_idx']
            self.transformers.append(TransformerInfo(is_categorical, ed-st))
            self.output_width += ed-st

def _loss_function(recon_x, x, sigmas, mu, logvar, transformers, factor):
    # Change output_info to transformers, which also contains type and dim information. 
    st = 0
    loss = []
    for transformer in transformers:
            if transformer.is_continuous:
                ed = st + transformer.output_width
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed
            elif transformer.is_categorical:
                ed = st + transformer.output_width
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

class TVAE(BaseSynthesizer):
  """
  """
  def __init__(self, data_info=None, embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        checkpoint_interval_seconds=None,
        **kwargs): 
    BaseSynthesizer.__init__(self, data_info=data_info, checkpoint_interval_seconds=checkpoint_interval_seconds, epochs=epochs, **kwargs)
    self.embedding_dim = embedding_dim
    self.compress_dims = compress_dims
    self.decompress_dims = decompress_dims

    self.l2scale = l2scale
    self.batch_size = batch_size
    self.loss_factor = loss_factor
    self._epochs = epochs
    
    # Set device - use base class method
    self.set_device()
    if not cuda or not torch.cuda.is_available():
        self.set_device(torch.device("cpu"))
    else:
        self.set_device(torch.device("cuda"))
        
    # Initialize transformer if data_info is provided
    if data_info is not None:
        self._transformer = TableTransformerInfo(data_info['transform_info'])
    else:
        self._transformer = None
  
  def _train(self, train_dataloader):
    """Train the TVAE model using tensor data from dataloader"""
    print("Device of TVAE model is:", self._device)
    
    self.init_model(train_dataloader)
    
    self.encoder.to(self._device)
    self.decoder.to(self._device)
    
    for i in range(self._epochs):
        self.current_epoch = i + 1
        if i % 100 == 0:
            print(f"Epoch {i}/{self._epochs}")
            
        for id_, data in enumerate(train_dataloader):
            self.optimizerAE.zero_grad()
            real = data.to(self._device)
            mu, std, logvar = self.encoder(real)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            rec, sigmas = self.decoder(emb)
            loss_1, loss_2 = _loss_function(
                rec, real, sigmas, mu, logvar,
                self._transformer.transformers, self.loss_factor
            )
            loss = loss_1 + loss_2
            loss.backward()
            self.optimizerAE.step()
            self.decoder.sigma.data.clamp_(0.01, 1.0)
            
            self.current_training_loss = loss.item()


  def _generate(self, n, condition=None):
        """Sample data similar to the training data.

        Args:
            n (int): Number of rows to sample.
            condition: Ignored for TVAE (not supported)

        Returns:
            torch.Tensor: Generated synthetic data
        """
        st = time.time()

        self.decoder.eval()

        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu())

        data = torch.cat(data, dim=0)
        data = data[:n]

        ed = time.time()
        print("Sampling time: ", ed - st)
        return data

  def init_model(self, train_dataloader):
        """Initialize data sampler, generator and synthesizers."""
        if self.model_loaded:
          return
        
        # Create transformer from data_info if needed
        if self._transformer is None and self.data_info is not None:
            self._transformer = TableTransformerInfo(self.data_info['transform_info'])
        
        data_dim = self._transformer.output_width
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        self.optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)
        
        self.model_loaded=True
        
  def get_state(self):
      """Write necessary model states into one dictionary."""
      state = {
        'transformer':self._transformer,
        'encoder': self.encoder.state_dict(),
        'decoder': self.decoder.state_dict(),
        'optimizerAE' : self.optimizerAE.state_dict(),
       }
      return state
        
  def load_state(self, checkpoint):
      """Load state from a file path/object"""
      state = torch.load(checkpoint)
      
      self._transformer = state['transformer']
      data_dim = self._transformer.output_width
      self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
      self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
      self.optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)
      
      self.encoder.load_state_dict(state['encoder']) 
      self.decoder.load_state_dict(state['decoder']) 
      self.optimizerAE.load_state_dict(state['optimizerAE']) 
      
      self.model_loaded = True

