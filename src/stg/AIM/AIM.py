"""CLI."""
import numpy as np
import torch
import pandas as pd
from torch import optim
import warnings
import time
import json
from scipy.optimize import bisect
import itertools
import argparse
from collections import defaultdict

from mbi import GraphicalModel, FactoredInference, Domain, Factor
from .mechanism import Mechanism
from .cdp2adp import cdp_rho
from .hdmm_matrix import Identity

from ..base import BaseSynthesizer

class Dataset:
    def __init__(self, df, domain, weights=None):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        assert weights is None or df.shape[0] == weights.size
        self.domain = domain
        self.df = df.loc[:,domain.attrs]
        self.weights = weights

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object 
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns = domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path, domain):
        """ Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        if isinstance(path, str):
          df = pd.read_csv(path)
        else:
          df = path
        if isinstance(domain, str):
          config = json.load(open(domain))
        else:
          config = domain
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)
    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:,cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)
    
    @property
    def records(self):
        return self.df.shape[0]

    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n+1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans
    


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl)&set(ax)) for ax in workload)
    return { cl : score(cl) for cl in downward_closure(workload) }

def filter_candidates(candidates, model, size_limit):
    ans = { }
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans



class AIM(Mechanism, BaseSynthesizer):
    """
    A Trusetics version of AIM model. See https://arxiv.org/pdf/2201.12677.pdf for details.
    """
    def __init__(self, data_info=None, epsilon=1.0, delta=1e-9, prng=np.random, rounds=None, max_model_size=80, structural_zeros={}, checkpoint_interval_seconds=30, epochs=None, **kwargs):
      Mechanism.__init__(self, epsilon, delta, prng)
      BaseSynthesizer.__init__(self, data_info=data_info, checkpoint_interval_seconds=checkpoint_interval_seconds, epochs=epochs, **kwargs)
      self.rounds = rounds
      self.max_model_size = max_model_size
      self.structural_zeros = structural_zeros
      self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
      self.prng = prng
   
    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt) 

        max_sensitivity = max(sensitivity.values()) # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def _train(self, train_dataloader):
        """Train the AIM model using tensor data from dataloader"""
        st = time.time()
        
        # Convert tensor dataloader to DataFrame for AIM processing
        all_data = []
        for batch in train_dataloader:
            all_data.append(batch.detach().cpu().numpy())
        
        train_data = np.concatenate(all_data, axis=0)
        train_data = pd.DataFrame(train_data)
        
        ed = time.time()
        print("Data preparation time is:", ed-st)
        
        # Convert data to AIM format
        data, W = self.prepare_parameters(train_data)
        
        # AIM algorithm
        rounds = self.rounds or 16*len(data.domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = { cl : data.project(cl).datavector() for cl in candidates }

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2*0.9*self.rho))
        epsilon = np.sqrt(8*0.1*self.rho/rounds)
       
        measurements = []
        print('Initial Sigma', sigma)
        rho_used = len(oneway)*0.5/sigma**2
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma,x.size)
            I = Identity(y.size) 
            measurements.append((I, y, sigma, cl))

        zeros = self.structural_zeros
        self.engine = FactoredInference(data.domain,iters=1000,warm_start=True,structural_zeros=zeros)
        self.model = self.engine.estimate(measurements)
        print("Model estimated!")
        t = 0
        terminate = False
        st = time.time()
        while not terminate:
            t += 1
            if t % 1000 == 0:
                print("Current t:",t)
            if self.rho - rho_used < 2*(0.5/sigma**2 + 1.0/8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2*0.9*remaining))
                epsilon = np.sqrt(8*0.1*remaining)
                terminate = True

            rho_used += 1.0/8 * epsilon**2 + 0.5/sigma**2
            size_limit = self.max_model_size*rho_used/self.rho

            small_candidates = filter_candidates(candidates, self.model, size_limit)
            cl = self.worst_approximated(small_candidates, answers, self.model, epsilon, sigma)

            n = data.domain.size(cl)
            Q = Identity(n) 
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = self.model.project(cl).datavector()

            self.model = self.engine.estimate(measurements)
            w = self.model.project(cl).datavector()
            print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w-z, 1) <= sigma*np.sqrt(2/np.pi)*n:
                print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma/2)
                sigma /= 2
                epsilon *= 2
        self.measurements = measurements
        ed = time.time()
        print("Model fitting time:", ed-st)

    def _generate(self, n, condition=None):
        """Sample data similar to the training data.

        Args:
            n (int): Number of rows to sample.
            condition: Ignored for AIM (not supported)

        Returns:
            torch.Tensor: Generated synthetic data
        """
        print('Generating Data...')
        self.engine.iters = 2500
        self.model = self.engine.estimate(self.measurements)
        synth = self.model.synthetic_data()

        # Return as tensor to match base class interface
        import torch
        data = synth.df.to_numpy()[:n]
        return torch.tensor(data, dtype=torch.float32)

    def init_model(self, train_data):
          if self.model_loaded:
            return
          
    def get_state(self):
        state = {
          'model':self.model,
         }
        return state
          
    def load_state(self, checkpoint):
        state = torch.load(checkpoint)
        
        self.model = state['model']
        
        self.model_loaded = True
        
    def default_params(self):
        """
        Return default parameters to run this program

        :returns: a dictionary of default parameter settings for each command line argument
        """
        params = {}
        params['dataset'] = '../data/adult.csv'
        params['domain'] = '../data/adult-domain.json'
        params['epsilon'] = 1.0
        params['delta'] = 1e-9
        params['noise'] = 'laplace'
        params['max_model_size'] = 80
        params['degree'] = 2
        params['num_marginals'] = None
        params['max_cells'] = 10000

        return params     
         
    def infer_domain(self,df):
        """
          Infer AIM domain parameter automatically based on input data.
        Args:
          df:input data
          
        Return:
          domain: a dictionay. Keys are column names and values are sizes of domain for each column. 
        """
        domain = {}
        print("Inferring domain for AIM!")
        int_mask = np.equal(np.mod(df.values, 1), 0).all(axis=0)
        # get a list of integer columns
        int_cols = df.columns[int_mask].tolist()
	
        for c in df.columns:
          # If this is a categorical / one-hot encoding column
          if c in int_cols:
              domain[c] = max(2,len(set(df[c])))
          # Else set as max value of this column + 1
          else:
              domain[c] = max(100, int(max(df[c])) + 1)
              
        return domain
         
    def prepare_parameters(self, train_data):
        params = self.default_params()
        domain = self.infer_domain(train_data)
        print("Dimension of transformed training data",train_data.shape)
        print(domain)
        data = Dataset.load(train_data, domain)

        workload = list(itertools.combinations(data.domain, params['degree']))
        workload = [cl for cl in workload if data.domain.size(cl) <= params['max_cells']]
        if params['num_marginals'] is not None:
            workload = [workload[i] for i in prng.choice(len(workload), params['num_marginals'], replace=False)]

        workload = [(cl, 1.0) for cl in workload]
        return data, workload
        #mech = AIM(params.epsilon, params.delta, max_model_size=params.max_model_size)
        #synth = mech.run(data, workload)

