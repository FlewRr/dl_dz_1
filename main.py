import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, AUROC
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from dataset import LoanDataset
from model import SimpleLoanModel, LoanModelExtra, LoanModelExtraWithBn, LoanModelExtraWithDropout
from train import train, test

if __name__ == "__main__":
    train_df = pd.read_csv("data/loan_test.csv").drop(columns=["id"])
    test_df = pd.read_csv("data/loan_train.csv").drop(columns=["id"])

    cat_features = train_df.drop(columns=["loan_status"]).select_dtypes("object").columns
    num_features = train_df.drop(columns=["loan_status"]).select_dtypes("number").columns

    CAT_MAPS = {
        cat_col:
            {value: encoded_value
             for encoded_value, value in enumerate(train_df[cat_col].unique())}
        for cat_col in cat_features
    }

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(scaler.fit_transform(train_df.drop(columns=["loan_status", *cat_features], axis=1)),
                            columns=num_features)
    Test_scaled = pd.DataFrame(scaler.transform(test_df.drop(columns=["loan_status", *cat_features], axis=1)),
                               columns=num_features)

    X_scaled = pd.concat([X_scaled, train_df[cat_features]], axis=1)
    Test_scaled = pd.concat([Test_scaled, test_df[cat_features]], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(X_scaled, train_df["loan_status"], test_size=0.2,
                                                      stratify=train_df["loan_status"])

    train_dataset = LoanDataset(x_train, y_train)
    val_dataset = LoanDataset(x_val, y_val)
    test_dataset = LoanDataset(Test_scaled, test_df["loan_status"])

    config = json.load(open("config.json"))

    ## Эксперимент 1
    print("Started experiment 1")
    model1 = SimpleLoanModel(config["experiment_1"]["hidden_size"])

    history = train(
        model=model1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config["experiment_1"]
    )

    model1.load_state_dict(torch.load("checkpoints/model.pt"))
    metrics1 = test(
        model=model1,
        test_dataset=test_dataset,
        config=config["experiment_1"])


    print("Experiment1: metrics ", metrics1)
    ## Эксперимент 2
    print("Started experiment 2")
    model2 = LoanModelExtra(config["experiment_2"]["hidden_size"],
                            config["experiment_2"]["num_blocks"])
    history2 = train(
        model=model2,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config["experiment_2"]
    )

    model2.load_state_dict(torch.load("checkpoints/model.pt"))
    metrics2 = test(
        model=model2,
        test_dataset=test_dataset,
        config=config["experiment_2"])
    print("Experiment2: metrics ", metrics2)

    ## Эксперимент 3
    print("Started experiment 3")
    model3 = LoanModelExtraWithBn(config["experiment_3"]["hidden_size"],
                                  config["experiment_3"]["num_blocks"])
    history3 = train(
        model=model3,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config["experiment_3"]
    )

    model3.load_state_dict(torch.load("checkpoints/model.pt"))
    metrics3 = test(
        model=model3,
        test_dataset=test_dataset,
        config=config["experiment_3"])
    print("Experiment3: metrics ", metrics3)

## Эксперимент 4
    print("Started experiment 4")
    model4 = LoanModelExtraWithDropout(config["experiment_4"]["hidden_size"],
                                      config["experiment_4"]["num_blocks"],
                                      config["experiment_4"]["p"])

    history4 = train(
        model=model4,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config["experiment_4"],
    )

    model4.load_state_dict(torch.load("checkpoints/model.pt"))

    metrics4 = test(
        model=model4,
        test_dataset=test_dataset,
        config=config["experiment_4"])

    print("Experiment4: metrics ", metrics4)


## Эксперимент 5
    print("Started experiment 5")
    model5 = LoanModelExtraWithDropout(config["experiment_5"]["hidden_size"],
                                      config["experiment_5"]["num_blocks"],
                                      config["experiment_5"]["p"])

    history5 = train(
        model=model5,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config["experiment_5"],
    )

    model5.load_state_dict(torch.load("checkpoints/model.pt"))

    metrics5 = test(
        model=model5,
        test_dataset=test_dataset,
        config=config["experiment_5"])

    print("Experiment5: metrics ", metrics5)
