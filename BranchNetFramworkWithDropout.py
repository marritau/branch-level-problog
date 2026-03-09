import torch
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
from beexai.utils.convert import convert_to_tensor
from tqdm import tqdm
from BranchNetWithDropout import BranchNetWithDropout

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, x_train, y_train, batch_size=32, shuffle=True):
        dataset = TabularDataset(x_train, y_train)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)


class BranchNetModelWithDropout(BranchNetWithDropout):
    """
    BranchNet with Dropout regularization and fit/predict interface.
    """
    def __init__(
            self,
            task: str = "classification",
            device=None,
            dtype=torch.float,
            dropout_rate: float = 0.3,  # Default dropout rate
    ) -> None:
        super().__init__(task=task, device=device, dtype=dtype, dropout_rate=dropout_rate)
        self.dropout_rate = dropout_rate

    def fit(
            self,
            x_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.Series, np.ndarray],
            x_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            learning_rate: float = 0.01,
            epochs: int = 100,
            batch_size: int = 32,
    ) -> None:
        """
        Fit the model with dropout regularization.
        
        Args:
            x_train: Training features
            y_train: Training labels
            x_val: Validation features (optional)
            y_val: Validation labels (optional)
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        x_train = convert_to_tensor(x_train, dtype=self.dtype, device=self.device)
        y_train = convert_to_tensor(y_train, dtype=torch.long, device=self.device)

        if x_val is not None and y_val is not None:
            x_val = convert_to_tensor(x_val, dtype=self.dtype, device=self.device)
            y_val = convert_to_tensor(y_val, dtype=torch.long, device=self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        dataloader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)

        self.train()
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            epoch_loss = 0
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            pbar.set_description(f"Loss: {avg_loss:.6f}")

            # Validation loss (optional)
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_output = self(x_val)
                    val_loss = criterion(val_output, y_val)
                    pbar.set_description(f"Loss: {avg_loss:.6f}, Val Loss: {val_loss.item():.6f}")
                self.train()

    def predict(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            x: Input features
            
        Returns:
            Predicted class labels
        """
        self.eval()
        x = convert_to_tensor(x, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self(x)
            predictions = torch.argmax(output, dim=1)
        return predictions.cpu().numpy()

    def predict_proba(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            x: Input features
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        x = convert_to_tensor(x, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self(x)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities.cpu().numpy()
