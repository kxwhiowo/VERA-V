import torch
from torch.utils.data import Dataset
import pandas as pd
import csv
import json
from sklearn.model_selection import train_test_split


class HelpfulHarmfulDataset(Dataset):
    def __init__(self, harmful_data, transform=None):
        self.data = harmful_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class StrongReject(Dataset):
    def __init__(self, api_load_fn, target_col):
        self.data = api_load_fn()
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][self.target_col]


def load(file_path, **kwargs):
    df = pd.read_csv(file_path)
    goals = df["goal"].tolist()
    targets = df["target"].tolist()
    return goals, targets

def load_harmbench(file_path, **kwargs):
    df = pd.read_csv(file_path)
    goals = df["Behavior"].tolist()
    BehaviorID = df["BehaviorID"].tolist()
    return goals, BehaviorID


def load_data(file_path):
    with open(file_path, "r") as file:
        print(data)
        data = file.readlines()
    return data

def load_hades(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    instructions = []
    categories = []
    
    for item in data:
        instructions.append(item["instruction"])
        categories.append(item["Big_category"])
    
    return instructions, categories

def load_illegal_activity(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    changed_questions = []
    phrase_types = []
    
    for key in sorted(data.keys(), key=int):
        item = data[key]
        changed_questions.append(item["Changed Question"])
        phrase_types.append(item["Phrase Type"])
    
    return changed_questions, phrase_types

def load_mm100(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    prompts = []
    categories = []
    
    for item in data['samples']:
        prompts.append(item["Changed Question"])
        categories.append(item["category"])
    
    return prompts, categories

def get_datasets(file_path, test_size):
    prompts, harmful_responses = load(file_path)
    harmful_data = list(zip(prompts, harmful_responses))
    harmful_train, harmful_test = train_test_split(harmful_data, test_size=test_size)
    harmful_train, harmful_val = train_test_split(harmful_train, test_size=test_size)

    train_dataset = HelpfulHarmfulDataset(harmful_train)
    test_dataset = HelpfulHarmfulDataset(harmful_test)
    val_dataset = HelpfulHarmfulDataset(harmful_val)

    return train_dataset, val_dataset, test_dataset
