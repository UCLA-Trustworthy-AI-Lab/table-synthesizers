from .models import Discriminator, Generator, Residual
from .data_sampler import DataSampler

import numpy as np
import torch
from torch import optim
import pandas as pd

from torch import optim
from torch.nn import functional

import warnings
import time

from packaging import version


from ..base import BaseSynthesizer

class TransformerInfo:
    def __init__(self,is_categorical, output_width) -> None:
        self.is_categorical = is_categorical
        self.output_width = output_width
        self.is_continuous = (not is_categorical)

class TableTransformerInfo:
    def __init__(self,transform_info) -> None:
        column_ranges_in_transformed = transform_info
        print(column_ranges_in_transformed)
        self.transformers = []
        self.output_width = 0
        for col_name, elements in column_ranges_in_transformed.items():
            is_categorical = elements['original_dtype'] == 'categorical'
            st,ed = elements['start_idx'], elements['end_idx']
            self.transformers.append(TransformerInfo(is_categorical, ed-st))
            self.output_width += ed-st

class CTGAN(BaseSynthesizer):
  """
  This model applies conditional generation and model-specific regularization to generated dataset with both categorical and continous columns. Implementation based on: https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/ctgan.py
  """
  def __init__(self, data_info, embedding_dim=128, generator_dim=(256,256), discriminator_dim=(256,256), generator_lr=0.0002, generator_decay=0.000001, discriminator_lr=0.0002, discriminator_decay=0.000001, batch_size=150, discriminator_steps=1, verbose=True, epochs=10, pac=5,checkpoint_interval_seconds=None,**kwarg):
    BaseSynthesizer.__init__(self, checkpoint_interval_seconds)
    self.model_loaded = False
    self._epochs = epochs
    #print(f"CTGAN epochs:{epochs}")
    self._transformer = TableTransformerInfo(data_info['transform_info'])
    #print("CTGAN transformer output width: ",self._transformer.output_width)
    #for t in self._transformer.transformers:
    #    print(t.output_width,t.is_categorical)

    # Initialization from original CTGAN
    assert batch_size % 2 == 0
    self._embedding_dim = embedding_dim
    self._generator_dim = generator_dim
    self._discriminator_dim = discriminator_dim

    self._generator_lr = generator_lr
    self._generator_decay = generator_decay
    self._discriminator_lr = discriminator_lr
    self._discriminator_decay = discriminator_decay

    self._batch_size = batch_size
    self._discriminator_steps = discriminator_steps
    self._verbose = verbose
    self._epochs = epochs
    self.pac = pac

    self._data_sampler = None
    self._generator = None
  
  def _train(
        self,
        train_dataloader
    ):
    """
        Trains the synthesizer model.

        This method initializes threading and sets the device for the model before beginning the training process. 
        It transforms the data and initializes the model. Training is conducted over several epochs, during which
        data is sampled and used to update the discriminator and generator in the GAN model.

        Args:
            train_dataloader (torch.dataloader): The training data to be used for model training.
            categorical_columns (list of str, optional): The list of categorical columns in the data.
            ordinal_columns (list of str, optional): The list of ordinal columns in the data.
            update_epsilon (float, optional): Epsilon value for differential privacy. If set, the method uses differentially private SGD.
            transformer (Transformer, optional): A transformer object to handle data transformations. If None, a new transformer is created.
            continuous_columns (list of str, optional): The list of continuous columns in the data.
            preprocessor_eps (float, optional): Privacy budget allocated for the transformation process. Default is 0.0.
            nullable (bool, optional): Flag to indicate whether the transformation process can handle null/none values. Default is False.

        Raises:
            ValueError: If NaN values are encountered in the transformed data.

        Returns:
            None
    """

    epochs = self._epochs

    self.init_model(train_dataloader)

    mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
    std = mean + 1

    steps_per_epoch = max(len(train_dataloader), 1)
    torch.autograd.set_detect_anomaly(True)

    for i in range(epochs):
        # Update the number of remaining epoch.
        # Needed for updating progress bar on website.
        self.current_epoch = i+1

        for id_ in range(steps_per_epoch):

            for n in range(self._discriminator_steps):
                #print(i, id_, n)
                fakez = torch.normal(mean=mean, std=std)

                condvec = self._data_sampler.sample_condvec(self._batch_size)
                #print("condvec sampled",condvec)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    #print("Real data with condvec is None:", real.shape)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    #print("col and opt:",col[perm], opt[perm])
                    real = self._data_sampler.sample_data(
                        self._batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                #print("Fakez input shape:",fakez.shape)
                fake = self._generator(fakez)
                #print("Fake after generator:", fake.shape)
                fakeact = self._apply_activate(fake)
                #print("fakeact after generator:", fakeact.shape)

                real = torch.from_numpy(real.astype('float32')).to(self._device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fakeact
                y_fake = self.discriminator(fake_cat)
                y_real = self.discriminator(real_cat)

                pen = self.discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self.pac)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                if torch.isnan(loss_d) or torch.isinf(loss_d):
                    print(-torch.mean(y_real), -torch.mean(y_fake), -torch.mean(fake_cat),-torch.mean(fake), -torch.mean(fakez))
                    raise ValueError("Invalid loss_d!!!")

                self.optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1)  # Clip gradients
                self.optimizerD.step()

            fakez = torch.normal(mean=mean, std=std)
            condvec = self._data_sampler.sample_condvec(self._batch_size)

            if condvec is None:
                c1, m1, col, opt = None, None, None, None
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)

            if c1 is not None:
                y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = self.discriminator(fakeact)

            if condvec is None:
                cross_entropy = 0
            else:
                cross_entropy = self._cond_loss(fake, c1, m1)

            loss_g = -torch.mean(y_fake) + cross_entropy
            if torch.isnan(loss_g) or torch.isinf(loss_g):
                print(-torch.mean(y_fake), torch.isnan(y_fake).any(), torch.isnan(fakeact).any(), torch.isnan(fake).any(), torch.isnan(fakez).any(), cross_entropy)
                raise ValueError("Invalid loss_g!!!")

            self.optimizerG.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self._generator.parameters(), max_norm=1)  # Clip gradients
            self.optimizerG.step()

            
        self.training_loss = loss_g

        if self._verbose:
            print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                  f'Loss D: {loss_d.detach().cpu(): .4f}',
                  flush=True)
                      

  def generate(self, n, condition_column=None, condition_value=None):
        """
        Sample data similar to the training data.
        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        st = time.time()
        if condition_column is not None and condition_value is not None:
            # Disable conditional generation function.
            raise NotImplementedError("Conditional generation in CTGAN has been disabled!")
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        ed = time.time()
        print("Sampling time: ", ed - st)
        return data

  def set_device(self):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
            
  def init_model(self, train_dataloader):
        """Initialize data sampler, generator and synthesizers."""
        if self.model_loaded:
          return
        
        self._data_sampler = DataSampler(
          train_dataloader,
          self._transformer.transformers,)

        data_dim = self._transformer.output_width
        print("Data_dim: ",data_dim)

        self._generator = Generator(
          self._embedding_dim + self._data_sampler.dim_cond_vec(),
          self._generator_dim,
          data_dim
          ).to(self._device)

        self.discriminator = Discriminator(
          data_dim + self._data_sampler.dim_cond_vec(),
          self._discriminator_dim,
          pac=self.pac
          ).to(self._device)

        self.optimizerG = optim.Adam(
          self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
          weight_decay=self._generator_decay
         )

        self.optimizerD = optim.Adam(
          self.discriminator.parameters(), lr=self._discriminator_lr,
          betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )
        
        self.model_loaded=True
        
  def get_state(self):
      """Write necessary model states into one dictionary."""
      state = {
        'data_sampler':self._data_sampler,
        'generator': self._generator.state_dict(),
        'discriminator': self.discriminator.state_dict(),
        'optimizerG_state' : self.optimizerG.state_dict(),
        'optimizerD_state' : self.optimizerD.state_dict(),
       }
      return state
        
  def load_state(self, checkpoint):
      """Load state from checkpoint"""
      state = torch.load(checkpoint)
      
      data_dim = self._transformer.output_width
      
      self._data_sampler = state['data_sampler']

      self._generator = Generator(
          self._embedding_dim + self._data_sampler.dim_cond_vec(),
          self._generator_dim,
          data_dim
          ).to(self._device)

      self.discriminator = Discriminator(
          data_dim + self._data_sampler.dim_cond_vec(),
          self._discriminator_dim,
          pac=self.pac
          ).to(self._device)

      self.optimizerG = optim.Adam(
          self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
          weight_decay=self._generator_decay
         )

      self.optimizerD = optim.Adam(
          self.discriminator.parameters(), lr=self._discriminator_lr,
          betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )
      
      self._generator.load_state_dict(state['generator']) 
      self.discriminator.load_state_dict(state['discriminator']) 
      self.optimizerG.load_state_dict(state['optimizerG_state']) 
      self.optimizerD.load_state_dict(state['optimizerD_state']) 
      
      self.model_loaded = True


        


  @staticmethod
  def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if torch.isnan(logits).any():
            raise ValueError("gumbel_softmax logits input has NaN!!!!")
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            print("Running other version!")
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
        if torch.isnan(transformed).any():
            warnings.warn(f"Nan found in gumbel_softmax! Standardizing logits.")
            print(torch.min(logits), torch.mean(logits), torch.max(logits), tau, hard, eps, dim)
            standardized_logits = (logits - torch.min(logits)) / (torch.max(logits) - torch.min(logits))
            transformed = functional.gumbel_softmax(standardized_logits, tau=tau, hard=hard, eps=eps, dim=dim)
        if torch.isnan(transformed).any():
            raise ValueError("gumbel_softmax returning NaN.")
        return transformed

  def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for transformer in self._transformer.transformers:
            if torch.isnan(data[:, st:(st + transformer.output_width)]).any():
                raise ValueError(f"Nan in {st}!")
            if transformer.is_continuous:
                ed = st + transformer.output_width
                transformed = torch.tanh(data[:, st:ed])
                data_t.append(transformed)
                st = ed
                if torch.isnan(transformed).any():
                    stt = 0
                    st_idx = ed-transformer.output_width
                    for transformer in self._transformer.transformers:
                        stt += transformer.output_width
                        print(stt)
                    raise ValueError(f"Nan in transformed numerical {st_idx}")
            elif transformer.is_categorical:
                ed = st + transformer.output_width
                transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed

        return torch.cat(data_t, dim=1)

  def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for t in self._transformer.transformers:
            if not t.is_categorical:
                # not discrete column
                st += t.output_width
            else:
                ed = st + t.output_width
                ed_c = st_c + t.output_width
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]