import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_dataset(data_path, has_v=True, has_a=True, has_t=True):
    dataset = np.load(data_path + "/metadata.npy", allow_pickle=True).item()

    dataset["warm_items"] = np.load(
        data_path + "/warm_items.npy", allow_pickle=True
    ).item()
    dataset["cold_items"] = np.load(
        data_path + "/cold_items.npy", allow_pickle=True
    ).item()

    dataset["train_data"] = np.load(
        data_path + "/train_interactions.npy", allow_pickle=True
    )

    dataset["val_data"] = np.load(
        data_path + "/val_interactions.npy", allow_pickle=True
    )
    dataset["val_warm_data"] = np.load(
        data_path + "/val_warm_interactions.npy", allow_pickle=True
    )
    dataset["val_cold_data"] = np.load(
        data_path + "/val_cold_interactions.npy", allow_pickle=True
    )

    dataset["test_data"] = np.load(
        data_path + "/test_interactions.npy", allow_pickle=True
    )
    dataset["test_warm_data"] = np.load(
        data_path + "/test_warm_interactions.npy", allow_pickle=True
    )
    dataset["test_cold_data"] = np.load(
        data_path + "/test_cold_interactions.npy", allow_pickle=True
    )

    dataset["t_feat"] = torch.from_numpy(np.load(data_path + "/t_features.npy")) if has_t else None
    dataset["a_feat"] = torch.from_numpy(np.load(data_path + "/a_features.npy")) if has_a else None
    dataset["v_feat"] = torch.from_numpy(np.load(data_path + "/v_features.npy")) if has_v else None

    return dataset


def preprocess_data(dataset):
    def preprocess_interactions(interactions, n_user):
        return {interaction[0]: interaction[1] + n_user for interaction in interactions}

    dataset["warm_items"] = set(
        item + dataset["n_users"] for item in dataset["warm_items"]
    )
    dataset["cold_items"] = set(
        item + dataset["n_users"] for item in dataset["cold_items"]
    )

    dataset["train_data"] = preprocess_interactions(
        dataset["train_data"], dataset["n_users"]
    )
    dataset["val_data"] = preprocess_interactions(
        dataset["val_data"], dataset["n_users"]
    )
    dataset["val_warm_data"] = preprocess_interactions(
        dataset["val_warm_data"], dataset["n_users"]
    )
    dataset["val_cold_data"] = preprocess_interactions(
        dataset["val_cold_data"], dataset["n_users"]
    )

    dataset["test_data"] = preprocess_interactions(
        dataset["test_data"], dataset["n_users"]
    )
    dataset["test_warm_data"] = preprocess_interactions(
        dataset["test_warm_data"], dataset["n_users"]
    )
    dataset["test_cold_data"] = preprocess_interactions(
        dataset["test_cold_data"], dataset["n_users"]
    )

    return dataset


def convert_interactions_to_user_item_dict(interactions, n_user):
    user_item_dict = {}
    for interaction in interactions:
        user = interaction[0]
        item = interaction[1]
        if user not in user_item_dict:
            user_item_dict[user] = [item]
        else:
            user_item_dict[user].append(item)

    for user in range(n_user) - user_item_dict.keys():
        user_item_dict[user] = []

    return user_item_dict


class TrainingDataset(Dataset):
    def __init__(
        self, num_user, num_item, user_item_dict, train_data, cold_set, num_neg
    ):
        self.train_data = train_data
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.cold_set = cold_set

        self.all_set = set(range(num_user, num_user + num_item)) - self.cold_set

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item = self.train_data[index]
        neg_item = random.sample(
            list(self.all_set - set(self.user_item_dict[user])), self.num_neg
        )

        user_tensor = torch.LongTensor([user] * (self.num_neg + 1))
        item_tensor = torch.LongTensor([pos_item] + neg_item)

        return user_tensor, item_tensor
