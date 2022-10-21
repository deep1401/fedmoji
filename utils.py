import copy
import numpy as np
from data import create_global_sharing_data_for_causalfedgsd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
      - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
      - clients (int): Number of Clients to split the data between

    returns:
      - Dictionary of image indexes for each client
    """

    num_items_per_client = int(len(dataset) / clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        client_dict[i] = set(
            np.random.choice(image_idxs, num_items_per_client, replace=False)
        )
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict


def iid_partition_causalfedgsd(dataset, clients, alpha=0.3):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
      - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
      - clients (int): Number of Clients to split the data between

    returns:
      - Dictionary of image indexes for each client
    """
    train_dataset, shared_dataset = create_global_sharing_data_for_causalfedgsd(
        dataset, size=0.3)
    num_items_per_client_train = int(len(train_dataset) / clients)
    num_items_per_client_shared = int(len(shared_dataset) * alpha)
    client_dict = {}
    # image_idxs = [i for i in range(len(train_dataset))]
    image_idxs = train_dataset.index.values.tolist()
    shared_idxs = shared_dataset.index.values.tolist()

    for i in range(clients):
        train_set = set(
            np.random.choice(
                image_idxs, num_items_per_client_train, replace=False)
        )
        shared_set = np.random.choice(
            shared_idxs, num_items_per_client_shared, replace=False)
        image_idxs = list(set(image_idxs) - train_set)
        train_set.update(shared_set)
        client_dict[i] = train_set
        # shared_idxs = list(set(shared_idxs) - client_dict[i])

    return client_dict


def non_iid_partition(
    dataset, clients, total_shards, shards_size, num_shards_per_client
):
    """
    non I.I.D parititioning of data over clients
    Sort the data by the label
    Divide the data into N shards of size S
    Each of the clients will get X shards

    params:
      - dataset (torch.utils.Dataset): Dataset containing the train_dataset
      - clients (int): Number of Clients to split the data between
      - total_shards (int): Number of shards to partition the data in
      - shards_size (int): Size of each shard
      - num_shards_per_client (int): Number of shards of size shards_size that each client receives

    returns:
      - Dictionary of image indexes for each client
    """

    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype="int64") for i in range(clients)}
    idxs = np.arange(len(dataset))
    # data_labels = dataset.targets.numpy()
    data_labels = np.array([elem[1] for elem in dataset])

    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(clients):
        rand_set = set(
            np.random.choice(shard_idxs, num_shards_per_client, replace=False)
        )
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate(
                (client_dict[i], idxs[rand *
                 shards_size: (rand + 1) * shards_size]),
                axis=0,
            )

    return client_dict


def pad_collate(batch):
    """Pads the sequence according the max length in a mini batch."""
    (xx, yy) = zip(*batch)
    x_lens = torch.tensor([len(x) for x in xx], dtype=torch.int64)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy), x_lens


class CustomDataset(Dataset):
    """Custom dataset wrapper."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        text, label = self.dataset[self.idxs[item]]
        return text, label
