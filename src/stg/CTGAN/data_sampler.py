import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class RandomIntegerDataset(Dataset):
    def __init__(self, num_samples, num_numerical_features, num_categorical_features, num_classes):
        self.num_samples = num_samples
        self.num_numerical_features = num_numerical_features
        self.num_categorical_features = num_categorical_features
        self.num_classes = num_classes

        self.numerical_data = torch.randn(num_samples, num_numerical_features)
        self.categorical_data = torch.randint(0, num_classes, (num_samples, num_categorical_features))
        self.one_hot_categorical_data = self._one_hot_encode(self.categorical_data, num_classes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        numerical_row = self.numerical_data[idx]
        categorical_row = self.one_hot_categorical_data[idx]
        return torch.cat((numerical_row, categorical_row), dim=0)

    def _one_hot_encode(self, data, num_classes):
        one_hot = torch.zeros(data.size(0), data.size(1) * num_classes)
        for i in range(data.size(1)):
            one_hot.scatter_(1, data[:, i].unsqueeze(1) + i * num_classes, 1)
        return one_hot

class SimpleTransformer:
    def __init__(self, output_width, is_categorical):
        self.output_width = output_width
        self.is_categorical = is_categorical
        self.output_start = 0  # Placeholder, this should be set properly

class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN.

    :param data_loader(DataLoader): The DataLoader to be conditionally sampled.
    :param transformers(list): The list of transformers for the model.
    :param discrete_column_category_prob(list): The category probabilities for each discrete column.
        Use this to pass in cached noisy probabilities. Data must match the schema of the original data.
    """

    def __init__(
            self,
            data_loader,
            transformers,
            *ignore,
            discrete_column_category_prob=None,
            **kwargs):
        self._data_loader = data_loader
        self._transformers = transformers

        self._per_column_scale = None
        self._discrete_column_category_prob = None

        n_discrete_columns = sum(
            [1 for t in self._transformers if t.is_categorical])

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype="int32")    

        # Prepare an interval matrix for efficiently sampling conditional vector
        max_category = max(
            [t.output_width for t in self._transformers
             if t.is_categorical], default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(
            n_discrete_columns, dtype='int32')
        self._discrete_column_category_freq = np.zeros(
            (n_discrete_columns, max_category))
        self._discrete_column_category_prob = np.zeros(
            (n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum(
            [t.output_width for t in self._transformers
             if t.is_categorical])
        
        num_rows = 0
        for batch in self._data_loader:
            st = 0
            current_cond_st = 0
            data = batch.numpy()
            current_id = 0
            for t in self._transformers:
                #print(current_id, t.output_width)
                if t.is_categorical:
                    ed = st + t.output_width
                    category_freq = np.sum(data[:, st:ed], axis=0)
                    category_freq = [1 if v < 1 else v for v in category_freq]
                    category_freq = np.array(category_freq, dtype='float64')
                    #category_prob = category_freq / np.sum(category_freq)
                    #print("Shape of self._discrete_column_category_prob:",self._discrete_column_category_prob.shape)
                    #print("category_prob:",category_prob)
                    self._discrete_column_category_freq[current_id, :t.output_width] += (
                        category_freq)
                    self._discrete_column_matrix_st[current_id] = st
                    self._discrete_column_cond_st[current_id] = current_cond_st
                    self._discrete_column_n_category[current_id] = t.output_width
                    current_cond_st += t.output_width
                    current_id += 1
                    num_rows += len(data)
                    st = ed
                else:
                    st += t.output_width

        self._discrete_column_category_prob = self._discrete_column_category_freq / num_rows

        if discrete_column_category_prob is not None:
            assert len(discrete_column_category_prob) == n_discrete_columns
            for i in range(n_discrete_columns):
                self._discrete_column_category_prob[i, :] = discrete_column_category_prob[i]
            self.total_spent = 0.0  # don't have to pay for cached noise

        self.num_rows = num_rows

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (self._discrete_column_cond_st[discrete_column_id]
                       + category_id_in_col)
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col
    
    def sample_original_condvec(self, batch_size):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch_size, self._n_categories), dtype='float32')
        batch_count = 0

        for batch in self._data_loader:
            data = batch.numpy()
            while batch_count < batch_size:
                row_idx = np.random.randint(0, data.shape[0])
                col_idx = np.random.randint(0, self._n_discrete_columns)
                matrix_st = self._discrete_column_matrix_st[col_idx]
                matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
                pick = np.argmax(data[row_idx, matrix_st:matrix_ed])
                cond[batch_count, pick + self._discrete_column_cond_st[col_idx]] = 1
                batch_count += 1
                if batch_count >= batch_size:
                    break

        return cond

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector using reservoir sampling.

        Returns:
            n rows of matrix data.
        """
        reservoir = []
        sampled_count = 0
        total_matching = 0

        for batch in self._data_loader:
            data = batch.numpy()
            for i in range(data.shape[0]):
                match = True
                if col is not None:
                    for c, o in zip(col, opt):
                        matrix_st = self._discrete_column_matrix_st[c]
                        matrix_ed = matrix_st + self._discrete_column_n_category[c]
                        #print("argmax output:",np.argmax(data[i, matrix_st:matrix_ed]), ". Target: ",o, ". matrix_st:",matrix_st)
                        if np.argmax(data[i, matrix_st:matrix_ed]) != o:
                            match = False
                            break
                if match:
                    total_matching += 1
                    if len(reservoir) < n:
                        reservoir.append(data[i])
                    else:
                        j = np.random.randint(0, total_matching)
                        if j < n:
                            reservoir[j] = data[i]
        return np.array(reservoir)
    
    def dim_cond_vec(self):
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id = self._discrete_column_matrix_st[condition_info["discrete_column_id"]
                                             ] + condition_info["value_id"]
        vec[:, id] = 1
        return vec
    
if __name__ == "__main__":
    # Create a dataset and dataloader
    num_samples = 1000
    num_numerical_features = 1
    num_categorical_features = 2
    num_classes = 4

    dataset = RandomIntegerDataset(num_samples, num_numerical_features, num_categorical_features, num_classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create transformers
    transformers = [
        SimpleTransformer(output_width=num_numerical_features, is_categorical=False),
    ]

    for i in range(num_categorical_features):
        transformers.append(SimpleTransformer(output_width=num_classes, is_categorical=True))

    # Instantiate DataSampler
    data_sampler = DataSampler(data_loader=dataloader, transformers=transformers)

    # Check discrete column probabilities
    print("Calculated Discrete Column Probabilities:")
    print(data_sampler._discrete_column_category_prob)

    # Calculate real probabilities
    all_data = torch.cat([batch for batch in dataloader], dim=0).numpy()
    start_idx = num_numerical_features
    end_idx = start_idx + num_categorical_features * num_classes 
    real_category_freq = np.sum(all_data[:, start_idx:end_idx], axis=0)
    real_category_prob = real_category_freq / np.sum(real_category_freq)

    print("\nReal Category Probabilities:")
    print(real_category_prob)
    print("\nCalculated Category Probabilities:")
    print(data_sampler._discrete_column_category_prob)


    # Test sample_condvec function
    batch_size = 5
    cond, mask, discrete_column_id, category_id_in_col = data_sampler.sample_condvec(batch_size)

    print("\nSampled Conditional Vector:")
    print(cond)
    print("Mask:")
    print(mask)
    print("Discrete Column ID:")
    print(discrete_column_id)
    print("Category ID in Column:")
    print(category_id_in_col)

    # Test sample_original_condvec function
    cond_original = data_sampler.sample_original_condvec(batch_size)
    print("\nSampled Original Conditional Vector:")
    print(cond_original)

    # Test sample_data function
    col = [0]
    opt = [2]
    sampled_data = data_sampler.sample_data(10, col, opt)
    print("\nSampled Data:")
    print(sampled_data)
    print("Check that sampled data matches given condition:")
    for row in sampled_data:
        assert np.argmax(row[start_idx:end_idx][col[0]*num_classes:(col[0]+1)*num_classes]) == opt[0], "Sampled data does not match the condition!"
    print("All sampled data matches the given condition.")