import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_dataset(data_path, has_v=True, has_a=True, has_t=True, device="cpu"):
    dataset = np.load(data_path + "/metadata.npy", allow_pickle=True).item()

    dataset["warm_items"] = np.load(
        data_path + "/warm_items.npy", allow_pickle=True
    ).item()
    dataset["cold_items"] = np.load(
        data_path + "/cold_items.npy", allow_pickle=True
    ).item()
    dataset['val_cold_items'] = np.load(
        data_path + "/val_cold_items.npy", allow_pickle=True
    ).item()
    dataset['test_cold_items'] = np.load(
        data_path + "/test_cold_items.npy", allow_pickle=True
    ).item()

    dataset["train_all_warm_data"] = np.load(
        data_path + "/train_all_warm_interactions.npy", allow_pickle=True
    )
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

    dataset["t_feat"] = torch.from_numpy(np.load(data_path + "/t_features.npy")).type(torch.float32).to(device) if has_t else None
    dataset["a_feat"] = torch.from_numpy(np.load(data_path + "/a_features.npy")).type(torch.float32).to(device) if has_a else None
    dataset["v_feat"] = torch.from_numpy(np.load(data_path + "/v_features.npy")).type(torch.float32).to(device) if has_v else None

    return dataset


def preprocess_data(dataset):
    def preprocess_interactions(interactions, n_users):
        interactions[:, 1] += n_users
        return interactions

    dataset["warm_items"] = np.array([
        item + dataset["n_users"] for item in dataset["warm_items"]
    ])
    dataset["cold_items"] = np.array([
        item + dataset["n_users"] for item in dataset["cold_items"]
    ])
    dataset['val_cold_items'] = np.array([
        item + dataset["n_users"] for item in dataset['val_cold_items']
    ])
    dataset['test_cold_items'] = np.array([
        item + dataset["n_users"] for item in dataset['test_cold_items']
    ])

    for key in ["train_all_warm_data", "train_data", "val_data", "val_warm_data", "val_cold_data", "test_data", "test_warm_data", "test_cold_data"]:
        dataset[key] = preprocess_interactions(dataset[key], dataset["n_users"])

    return dataset


def convert_interactions_to_user_item_dict(interactions, n_users):
    user_item_dict = {user_id: [] for user_id in range(n_users)}

    for interaction in interactions:
        user_item_dict[interaction[0]].append(interaction[1])

    return user_item_dict


def convert_interactions_to_user_item_list(interactions, n_users):
    user_item_dict = convert_interactions_to_user_item_dict(interactions, n_users)

    user_item_list = []
    for user_id, list_items in user_item_dict.items():
        if len(list_items) == 0:
            continue
        user_item_list.append([user_id] + list_items)

    return user_item_list


class TrainingDataset(Dataset):
    def __init__(
        self, num_user, num_item, user_item_dict, train_data, cold_set, num_neg, device="cpu"
    ):
        self.train_data = train_data
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.cold_set = set(cold_set)
        self.device = device

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
