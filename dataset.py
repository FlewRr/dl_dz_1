import torch
from torch.utils.data import Dataset
import pandas as pd


cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade',
       'cb_person_default_on_file']

num_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

CAT_MAPS = {'person_home_ownership': {'MORTGAGE': 0, 'RENT': 1, 'OWN': 2, 'OTHER': 3},
 'loan_intent': {'EDUCATION': 0,
  'HOMEIMPROVEMENT': 1,
  'MEDICAL': 2,
  'DEBTCONSOLIDATION': 3,
  'VENTURE': 4,
  'PERSONAL': 5},
 'loan_grade': {'A': 0, 'D': 1, 'B': 2, 'C': 3, 'F': 4, 'E': 5, 'G': 6},
 'cb_person_default_on_file': {'N': 0, 'Y': 1}}

class LoanDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        target_item = self.target.iloc[idx]

        out = {"num_features": {}, "cat_features": {}}

        for col in item.index.to_list():
            if col in cat_features:
                out["cat_features"][col] = torch.scalar_tensor(CAT_MAPS[col][item[col]], dtype=torch.long)
            else:
                out["num_features"][col] = torch.scalar_tensor(item[col], dtype=torch.float)

        out["target"] = torch.scalar_tensor(target_item, dtype=torch.float)

        return out

class LoanCollator:
    def __call__(self, items):
        out = {"num_features": {}, "cat_features": {}}

        if "target" in items[0]:
            out["target"] = torch.stack([x["target"] for x in items])

        for key in items[0]["cat_features"].keys():
            out["cat_features"][key] = torch.stack([x["cat_features"][key] for x in items])

        for key in items[0]["num_features"].keys():
            out["num_features"][key] = torch.stack([x["num_features"][key] for x in items])

        return out