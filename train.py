import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, AUROC
from tqdm import tqdm
from dataset import LoanDataset, LoanCollator


def train(model: nn.Module, train_dataset: Dataset, eval_dataset: Dataset, config: dict):
    seed = 52
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = config["num_epochs"]
    lr = config["lr"]
    weight_decay = config["decay"] if "decay" in config else 0.0
    batch_size = config["batch_size"]

    torch.random.manual_seed(seed)

    model = model.to(device)
    loss_bce = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    eval_dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    history = {"train_losses": [], "train_auc": [], "val_losses": [], "val_auc": []}
    best_eval_loss = float('inf')

    for i_epoch in tqdm(range(n_epochs)):
        train_loss = MeanMetric().to(device)
        train_rocauc = AUROC(task='binary').to(device)
        for i, batch in enumerate(train_dl):
            # data, target = batch[0].to(device), batch[1].to(device)
            # result = model(data)

            ##
            num_features, cat_features, target = batch["num_features"], batch["cat_features"], batch["target"].to(
                device)

            for col in num_features:
                num_features[col] = num_features[col].to(device)

            for col in cat_features:
                cat_features[col] = cat_features[col].to(device)

            result = model(num_features=num_features, cat_features=cat_features)

            ##

            loss_value = loss_bce(result, target)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss_value)
            train_rocauc.update(torch.sigmoid(result), target)

        train_loss = train_loss.compute().item()
        train_rocauc = train_rocauc.compute().item()

        history["train_losses"].append(train_loss)
        history["train_auc"].append(train_rocauc)

        eval_loss = MeanMetric().to(device)
        eval_rocauc = AUROC(task='binary').to(device)

        model.eval()
        with torch.no_grad():
            for i_eval, batch_eval in enumerate(eval_dl):
                # data, target = batch_eval[0].to(device), batch_eval[1].to(device)

                # result_eval = model(data)

                ##
                num_features, cat_features, target = batch_eval["num_features"], batch_eval["cat_features"], batch_eval[
                    "target"].to(device)

                for col in num_features:
                    num_features[col] = num_features[col].to(device)

                for col in cat_features:
                    cat_features[col] = cat_features[col].to(device)

                result_eval = model(num_features=num_features, cat_features=cat_features)

                ##

                eval_loss_value = loss_bce(result_eval, target)

                eval_loss.update(eval_loss_value)
                eval_rocauc.update(torch.sigmoid(result_eval), target)
        model.train()

        eval_loss = eval_loss.compute().item()
        eval_rocauc = eval_rocauc.compute().item()

        history["val_losses"].append(eval_loss)
        history["val_auc"].append(eval_rocauc)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), "checkpoints/model.pt")

    return history


def test(model: nn.Module, test_dataset: Dataset, config: dict):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    test_dl = DataLoader(test_dataset, batch_size=config["batch_size"])

    loss_bce = torch.nn.BCEWithLogitsLoss()
    test_loss = MeanMetric().to(device)
    test_rocauc = AUROC(task='binary').to(device)

    model.eval()
    with torch.no_grad():
        for i_eval, batch in enumerate(test_dl):
            # data, target = batch[0].to(device), batch[1].to(device)

            # result_eval = model(data)

            ##
            num_features, cat_features, target = batch["num_features"], batch["cat_features"], batch["target"].to(
                device)

            for col in num_features:
                num_features[col] = num_features[col].to(device)

            for col in cat_features:
                cat_features[col] = cat_features[col].to(device)

            result_eval = model(num_features=num_features, cat_features=cat_features)

            ##

            loss_value = loss_bce(result_eval, target)

            test_loss.update(loss_value)
            test_rocauc.update(torch.sigmoid(result_eval), target)

    test_loss = test_loss.compute().item()
    test_rocauc = test_rocauc.compute().item()

    return test_loss, test_rocauc