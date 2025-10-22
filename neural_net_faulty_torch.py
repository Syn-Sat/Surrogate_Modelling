# Surrogate Model Pipeline for Vapour Pressure of Hydrocarbons (with H,C,N molecules) (CSV-based)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from rdkit import RDLogger
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog('rdApp.warning')

pandarallel.initialize(progress_bar=False, verbose=0)

def load_csv_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return pd.DataFrame()

def compute_descriptors(inchi):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def mol_from_inchi_cached(inchi):
        return Chem.MolFromInchi(inchi)

    mol = mol_from_inchi_cached(inchi)

    def num_branches(inchi):
        return sum((atom.GetAtomicNum() == 6 and atom.GetDegree() > 2) for atom in mol.GetAtoms())
    
    def num_nitrogens(inchi):
        return sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms())

    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumBranches': num_branches(inchi),
            'NumNitrogens': num_nitrogens(inchi)
        }
    else:
        return {
            'MolWt': np.nan,
            'TPSA': np.nan,
            'NumHDonors': np.nan,
            'NumHAcceptors': np.nan,
            'MolLogP': np.nan,
            'NumBranches': np.nan,
            'NumNitrogens': np.nan
        }

def is_alkaneetc(inchi):
    from rdkit import Chem
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in [1, 6, 7]: #H,C,N
            return False
    return True

def train_neural_net(df):

    desc_df = df['InChI'].parallel_apply(compute_descriptors).parallel_apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1).dropna()

    features = ['Temperature_K', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP', 'NumBranches', 'NumNitrogens']
    X = df[features]
    y = np.log(df['VapourPressure_kPa']).values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # regression output
            )
        def forward(self, x):
            return self.net(x)

    model = SimpleNN(input_dim=8).to(device)

    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Step 7: Training loop ---
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}], Loss: {loss.item():.4f}")

    # --- Step 8: Evaluate on test set ---
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mse = criterion(y_pred, y_test_tensor).item()
        print(f"\nTest RMSE (kPa): {np.exp(np.sqrt(mse)):.4f}")

    return model

if __name__ == "__main__":
    csv_file = "vapour_pressure_data_final.csv"
    df = load_csv_data(csv_file)

    if not df.empty:
        df = df[df['InChI'].parallel_apply(is_alkaneetc)]

        print(f"Filtered {len(df)} cleaned hydrocarbon records with vapor pressure data.")

        if not df.empty:
            trained_model = train_neural_net(df)
        else:
            print("No vapor pressure data after filtering.")
    else:
        print("No CSV data.")