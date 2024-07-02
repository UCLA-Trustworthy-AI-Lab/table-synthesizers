
import math
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.utils.data
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.autograd import Variable

from packaging import version
from torch.nn import functional
from torch.utils.data import DataLoader, Subset



from ..CTGAN.data_sampler import DataSampler

from .privacy_utils import weights_init, pate, moments_acc

from ..base import BaseSynthesizer


class Discriminator(Module):
    def __init__(self, input_dim, discriminator_dim, loss, pac=10):
        super(Discriminator, self).__init__()
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)

        dim = input_dim * pac
        #  print ('now dim is {}'.format(dim))
        self.pac = pac
        self.pacdim = dim

        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        if loss == "cross_entropy":
            seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def dragan_penalty(self, real_data, device="cpu", pac=10, lambda_=10):
        # real_data = torch.from_numpy(real_data).to(device)
        alpha = (
            torch.rand(real_data.shape[0], 1, device=device)
            .squeeze()
            .expand(real_data.shape[0])
        )
        delta = torch.normal(
            mean=0.0, std=float(pac), size=real_data.shape, device=device
        )  # 0.5 * real_data.std() * torch.rand(real_data.shape)
        x_hat = Variable(
            (alpha * real_data.T + (1 - alpha) * (real_data + delta).T).T,
            requires_grad=True,
        )

        pred_hat = self(x_hat.float())

        gradients = torch.autograd.grad(
            outputs=pred_hat,
            inputs=x_hat,
            grad_outputs=torch.ones(pred_hat.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        dragan_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return dragan_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

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

def split_dataloader(dataloader, num_splits):
    dataset = dataloader.dataset
    total_size = len(dataset)
    split_size = total_size // num_splits
    remainder = total_size % num_splits

    indices = list(range(total_size))
    subsets = []

    start_idx = 0
    for i in range(num_splits):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        subset_indices = indices[start_idx:end_idx]
        subsets.append(Subset(dataset, subset_indices))
        start_idx = end_idx

    dataloaders = [DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True) for subset in subsets]
    
    return dataloaders

class PATECTGAN(BaseSynthesizer):
    """
        CTGAN with PATE framework applied to enable differential privacy. Based on: https://github.com/opendp/smartnoise-sdk/blob/main/synth/snsynth/pytorch/nn/patectgan.py
    """
    def __init__(
        self,
        data_info, 
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        verbose=True,
        epochs=300,
        pac=1,
        cuda=True,
        epsilon=3,
        binary=False,
        regularization='dragan',
        loss="cross_entropy",
        teacher_iters=5,
        student_iters=5,
        sample_per_teacher=1000,
        delta=None,
        noise_multiplier=1e-3,
        moments_order=100,
        checkpoint_interval_seconds = 30,
        **kwarg,
    ):
        BaseSynthesizer.__init__(self, checkpoint_interval_seconds)
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
        self.epsilon = epsilon
        self.verbose = verbose
        self.loss = loss

        # PATE params
        self.regularization = regularization if self.loss != "wasserstein" else "dragan"
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters
        self.pd_cols = None
        self.pd_index = None
        self.binary = binary
        self.sample_per_teacher = sample_per_teacher
        self.noise_multiplier = noise_multiplier
        self.moments_order = moments_order
        self.delta = delta

        self._transformer = TableTransformerInfo(data_info['transform_info'])

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

    def train(
        self,
        train_dataloader
    ):
        #self.start_threading()

        self._batch_size = min(self._batch_size, len(train_dataloader.dataset))

        sample_per_teacher = (
            self.sample_per_teacher if self.sample_per_teacher < len(train_dataloader.dataset) else 1000
        )
        self.num_teachers = int(len(train_dataloader.dataset) / sample_per_teacher) + 1

        data_partitions = split_dataloader(train_dataloader, self.num_teachers)

        data_dim = self._transformer.output_width

        self.cond_generator = DataSampler(
            train_dataloader,
            self._transformer.transformers
        )

        cached_probs = self.cond_generator._discrete_column_category_prob

        cond_generator = [
            DataSampler(
                d,
                self._transformer.transformers,
                discrete_column_category_prob=cached_probs,
            )
            for d in data_partitions
        ]

        self._generator = Generator(
            self._embedding_dim + self.cond_generator.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        self.discriminator = Discriminator(
            data_dim + self.cond_generator.dim_cond_vec(),
            self._discriminator_dim,
            self.loss,
            self.pac,
        ).to(self._device)

        self.student_disc = self.discriminator
        self.student_disc.apply(weights_init)

        self.teacher_disc = [self.discriminator for i in range(self.num_teachers)]
        for i in range(self.num_teachers):
            self.teacher_disc[i].apply(weights_init)

        self.optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        self.optimizer_s = optim.Adam(self.student_disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.optimizer_t = [
            optim.Adam(
                self.teacher_disc[i].parameters(),
                lr=self._discriminator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._discriminator_decay,
            )
            for i in range(self.num_teachers)
        ]

        noise_multiplier = self.noise_multiplier
        alphas = torch.tensor(
            [0.0 for i in range(self.moments_order)], device=self._device
        )
        l_list = 1 + torch.tensor(range(self.moments_order), device=self._device)
        eps = torch.zeros(1)

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        real_label = 1
        fake_label = 0

        criterion = nn.BCELoss() if (self.loss == "cross_entropy") else self.w_loss

        if self.verbose:
            print(
                "using loss {} and regularization {}".format(
                    self.loss, self.regularization
                )
            )

        iteration = 0

        if self.delta is None:
            self.delta = 1 / (len(train_dataloader.dataset) * np.sqrt(len(train_dataloader.dataset)))

        while eps.item() < self.epsilon:
            iteration += 1

            eps = min((alphas - math.log(self.delta)) / l_list)

            if eps.item() > self.epsilon:
                if iteration == 1:
                    raise ValueError(
                        "Inputted epsilon parameter is too small to"
                        + " create a private dataset. Try increasing epsilon and rerunning."
                    )
                break

            # train teacher discriminators
            for t_2 in range(self.teacher_iters):
                for i in range(self.num_teachers):
                    partition_data = data_partitions[i]
                    data_sampler = DataSampler(
                        partition_data,
                        self._transformer.transformers,
                        discrete_column_category_prob=cached_probs,
                    )
                    fakez = torch.normal(mean, std=std).to(self._device)

                    condvec = cond_generator[i].sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fake

                    self.optimizer_t[i].zero_grad()

                    y_all = torch.cat(
                        [self.teacher_disc[i](fake_cat), self.teacher_disc[i](real_cat)]
                    )
                    label_fake = torch.full(
                        (int(self._batch_size / self.pac), 1),
                        fake_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    label_true = torch.full(
                        (int(self._batch_size / self.pac), 1),
                        real_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    labels = torch.cat([label_fake, label_true])

                    error_d = criterion(y_all.squeeze(), labels.squeeze())
                    error_d.backward()

                    if self.regularization == "dragan":
                        pen = self.teacher_disc[i].dragan_penalty(
                            real_cat, device=self._device
                        )
                        pen.backward(retain_graph=True)

                    self.optimizer_t[i].step()
            ###
            # train student discriminator
            for t_3 in range(self.student_iters):
                data_sampler = DataSampler(
                    train_dataloader,
                    self._transformer.transformers,
                    discrete_column_category_prob=cached_probs,
                )
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample_data(self._batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample_data(
                        self._batch_size, col[perm], opt[perm]
                    )
                    c2 = c1[perm]

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                else:
                    fake_cat = fakeact

                fake_data = fake_cat

                ###
                predictions, votes = pate(
                    fake_data, self.teacher_disc, noise_multiplier, device=self._device
                )

                output = self.student_disc(fake_data.detach())

                # update moments accountant
                alphas = alphas + moments_acc(
                    self.num_teachers,
                    votes,
                    noise_multiplier,
                    l_list,
                    device=self._device,
                )

                loss_s = criterion(
                    output.squeeze(), predictions.float().to(self._device).squeeze()
                )

                self.optimizer_s.zero_grad()
                loss_s.backward()

                if self.regularization == "dragan":
                    vals = torch.cat([predictions, fake_data], axis=1)
                    ordered = vals[vals[:, 0].sort()[1]]
                    data_list = torch.split(
                        ordered, predictions.shape[0] - int(predictions.sum().item())
                    )
                    synth_cat = torch.cat(data_list[1:], axis=0)[:, 1:]
                    pen = self.student_disc.dragan_penalty(synth_cat, device=self._device)
                    pen.backward(retain_graph=True)

                self.optimizer_s.step()

                # print ('iterator {i}, student discriminator loss is {j}'.format(i=t_3, j=loss_s))

            # train generator
            fakez = torch.normal(mean=mean, std=std)
            condvec = self.cond_generator.sample_condvec(self._batch_size)

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
                y_fake = self.student_disc(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = self.student_disc(fakeact)

            if condvec is None:
                cross_entropy = 0
            else:
                cross_entropy = self._cond_loss(fake, c1, m1)

            if self.loss == "cross_entropy":
                label_g = torch.full(
                    (int(self._batch_size / self.pac), 1),
                    real_label,
                    dtype=torch.float,
                    device=self._device,
                )
                loss_g = criterion(y_fake.squeeze(), label_g.float().squeeze())
                loss_g = loss_g + cross_entropy
            else:
                loss_g = -torch.mean(y_fake) + cross_entropy

            self.optimizerG.zero_grad()
            loss_g.backward()
            
            self.training_loss = loss_g
            self.optimizerG.step()

            if self.verbose:
                print(
                    "eps: {:f} \t G: {:f} \t D: {:f}".format(
                        eps, loss_g.detach().cpu(), loss_s.detach().cpu()
                    )
                )
        #self.stop_threading()

    def w_loss(self, output, labels):
        vals = torch.cat([labels[None, :], output[None, :]], axis=1)
        ordered = vals[vals[:, 0].sort()[1]]
        data_list = torch.split(ordered, labels.shape[0] - int(labels.sum().item()), dim=1)
        fake_score = data_list[0][:, 1]
        true_score = torch.cat(data_list[1:], axis=0)[:, 1]
        w_loss = -(torch.mean(true_score) - torch.mean(fake_score))
        return w_loss

    def generate(self, n, condition_column=None, condition_value=None):
        """
        TODO: Add condition_column support 
        """
        self._generator.eval()

        # output_info = self._transformer.output_info
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self.cond_generator.sample_original_condvec(self._batch_size)

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

        return data[:n]

    def fit(self, data, *ignore, transformer=None, categorical_columns=[], ordinal_columns=[], continuous_columns=[], preprocessor_eps=0.0, nullable=False):
        self.train(data)

    def sample(self, n_samples):
        return self.generate(n_samples)

            
    def get_state(self):
        state = {
          'cond_generator':self.cond_generator,
          'transformer':self._transformer,
          'generator': self._generator.state_dict(),
          'student_disc': self.student_disc.state_dict(),
          'teacher_disc': {},
          'num_teachers':self.num_teachers,
          'optimizerG_state' : self.optimizerG.state_dict(),
          'optimizer_s_state' : self.optimizer_s.state_dict(),
          'optimizer_t_state' : {}
         }
         
        for i in range(self.num_teachers):
          state['teacher_disc'][i] = self.teacher_disc[i].state_dict()
          state['optimizer_t_state'][i] = self.optimizer_t[i].state_dict()
            
        return state
          
    def load_state(self, checkpoint):
        state = torch.load(checkpoint)
        
        self._transformer = state['transformer']
        data_dim = self._transformer.output_width
        
        self.cond_generator = state['cond_generator']

        self._generator = Generator(
            self._embedding_dim + self.cond_generator.dim_cond_vec(),
            self._generator_dim,
            data_dim
            ).to(self._device)

        self.discriminator = Discriminator(
            data_dim + self.cond_generator.dim_cond_vec(),
            self._discriminator_dim,
            self.loss,
            pac=self.pac
            ).to(self._device)

        self.student_disc = self.discriminator
        self.num_teachers = state['num_teachers']

        self.teacher_disc = [self.discriminator for i in range(self.num_teachers)]

        self.optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        self.optimizer_s = optim.Adam(self.student_disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.optimizer_t = [
            optim.Adam(
                self.teacher_disc[i].parameters(),
                lr=self._discriminator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._discriminator_decay,
            )
            for i in range(self.num_teachers)
        ]
        
        self._generator.load_state_dict(state['generator']) 
        self.student_disc.load_state_dict(state['student_disc'])
        self.optimizerG.load_state_dict(state['optimizerG_state']) 
        self.optimizer_s.load_state_dict(state['optimizer_s_state']) 
        for i in range(self.num_teachers):
            self.teacher_disc[i].load_state_dict(state['teacher_disc'][i]) 
            self.optimizer_t[i].load_state_dict(state['optimizer_t_state'][i]) 
        
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
