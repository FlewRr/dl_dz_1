import torch
from torch import nn, Tensor


## Model for Experiment 1
class SimpleLoanModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.emb1 = nn.Embedding(4, hidden_size // 8)
        self.emb2 = nn.Embedding(6, hidden_size // 8)
        self.emb3 = nn.Embedding(7, hidden_size // 8)
        self.emb4 = nn.Embedding(2, hidden_size // 8)

        self.linear_in = nn.Linear(7, hidden_size // 2)

        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

        self.linear_out = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, num_features: Tensor, cat_features: Tensor):
        x_1 = self.emb1(cat_features["person_home_ownership"])
        x_2 = self.emb2(cat_features["loan_intent"])
        x_3 = self.emb3(cat_features["loan_grade"])
        x_4 = self.emb4(cat_features["cb_person_default_on_file"])

        x_num = self.linear_in(torch.stack([feature for feature in num_features.values()], dim=-1))

        x = torch.cat([x_1, x_2, x_3, x_4, x_num], axis=-1)

        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear_out(x)
        return x.squeeze(-1)


## Model for Experiment 2
class LoanModelBlock2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class LoanModelExtra(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int = 1):
        super().__init__()
        self.emb1 = nn.Embedding(4, hidden_size // 8)
        self.emb2 = nn.Embedding(6, hidden_size // 8)
        self.emb3 = nn.Embedding(7, hidden_size // 8)
        self.emb4 = nn.Embedding(2, hidden_size // 8)

        self.numeric_linear = nn.Linear(7, hidden_size // 2)

        self.linear_blocks = nn.Sequential(*[LoanModelBlock2(hidden_size) for _ in range(num_blocks)])

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, num_features: Tensor, cat_features: Tensor):
        x_1 = self.emb1(cat_features["person_home_ownership"])
        x_2 = self.emb2(cat_features["loan_intent"])
        x_3 = self.emb3(cat_features["loan_grade"])
        x_4 = self.emb4(cat_features["cb_person_default_on_file"])

        x_num = self.numeric_linear(torch.stack([feature for feature in num_features.values()], dim=-1))

        x = torch.cat([x_1, x_2, x_3, x_4, x_num], axis=-1)

        for block in self.linear_blocks:
            x = block(x)

        result = self.linear_out(x)

        return result.squeeze(-1)


# Model for Experiment 3
class LoanModelBlock3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        to_skip = x
        x = self.bn(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x + to_skip


class LoanModelExtraWithBn(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int = 1):
        super().__init__()
        self.emb1 = nn.Embedding(4, hidden_size // 8)
        self.emb2 = nn.Embedding(6, hidden_size // 8)
        self.emb3 = nn.Embedding(7, hidden_size // 8)
        self.emb4 = nn.Embedding(2, hidden_size // 8)

        self.numeric_linear = nn.Linear(7, hidden_size // 2)

        layers = []

        for _ in range(num_blocks):
            layers.append(LoanModelBlock3(hidden_size))

        self.linear_blocks = nn.Sequential(*layers)

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, num_features: Tensor, cat_features: Tensor) -> Tensor:
        x_1 = self.emb1(cat_features["person_home_ownership"])
        x_2 = self.emb2(cat_features["loan_intent"])
        x_3 = self.emb3(cat_features["loan_grade"])
        x_4 = self.emb4(cat_features["cb_person_default_on_file"])

        x_num = self.numeric_linear(torch.stack([feature for feature in num_features.values()], dim=-1))

        x = torch.cat([x_1, x_2, x_3, x_4, x_num], axis=-1)

        x = self.linear_blocks(x)

        result = self.linear_out(x)

        return result.view(-1)

# Model for Experiment 4, 5
class LoanModelBlock4(nn.Module):
    def __init__(self, hidden_size, p):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        to_skip = x
        x = self.bn(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(self.relu(self.linear2(x)))

        return x + to_skip


class LoanModelExtraWithDropout(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int = 1, p: int = 0.5):
        super().__init__()
        self.emb1 = nn.Embedding(4, hidden_size)
        self.emb2 = nn.Embedding(6, hidden_size)
        self.emb3 = nn.Embedding(7, hidden_size)
        self.emb4 = nn.Embedding(2, hidden_size)

        self.numeric_linear = nn.Linear(7, hidden_size)

        layers = []

        for _ in range(num_blocks):
            layers.append(LoanModelBlock4(hidden_size, p))

        self.linear_blocks = nn.Sequential(*layers)

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, num_features: Tensor, cat_features: Tensor) -> Tensor:
        x_1 = self.emb1(cat_features["person_home_ownership"])
        x_2 = self.emb2(cat_features["loan_intent"])
        x_3 = self.emb3(cat_features["loan_grade"])
        x_4 = self.emb4(cat_features["cb_person_default_on_file"])

        x_num = self.numeric_linear(torch.stack([feature for feature in num_features.values()], dim=-1))

        x = x_1 + x_2 + x_3 + x_4 + x_num

        x = self.linear_blocks(x)

        result = self.linear_out(x)

        return result.view(-1)