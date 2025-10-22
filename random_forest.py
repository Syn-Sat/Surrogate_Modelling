# Surrogate Model Pipeline for Vapour Pressure of Hydrocarbons (with H,C,N molecules) (CSV-based)

import pandas as pd
import numpy as np
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pandarallel import pandarallel

RDLogger.DisableLog('rdApp.warning')

pandarallel.initialize(progress_bar=True, nb_workers=12)

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

def train_random_forest(df):

    desc_df = df['InChI'].parallel_apply(compute_descriptors).parallel_apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1).dropna()

    features = ['Temperature_K', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP', 'NumBranches', 'NumNitrogens']
    X = df[features]
    y = np.log(df['VapourPressure_kPa'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = RandomForestRegressor(
        n_estimators=150,      
        max_depth=None,               
        n_jobs=-1              
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred))  
    mape = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))
    r2 = r2_score(y_test, y_pred)

    plt.scatter(y_test, y_pred, color='blue')
    m, b = np.polyfit(y_test, y_pred, 1)

    plt.plot(y_test, m*y_test + b, 'r--', label=f'Fit: y={m:.2f}x+{b:.2f}')

    plt.xlabel('Experimental')
    plt.ylabel('Predicted')

    print("\nRandom Forest Model Evaluation:")
    print("MAE (kPa):", mae)
    print("MAPE: ", mape)
    print("R^2 Score:", r2)

    plt.legend()
    plt.show()

    return model

if __name__ == "__main__":
    csv_file = "vapour_pressure_data_final.csv"
    df = load_csv_data(csv_file)

    if not df.empty:
        df = df[df['InChI'].parallel_apply(is_alkaneetc)]

        print(f"Filtered {len(df)} cleaned hydrocarbon records with vapor pressure data.")

        print("Min P (kPa):", df['VapourPressure_kPa'].min())
        print("Max P (kPa):", df['VapourPressure_kPa'].max())
        print("Min T (K)", df['Temperature_K'].min)
        print("Max T (K)", df['Temperature_K'].max)
        if not df.empty:
            trained_model = train_random_forest(df)
        else:
            print("No vapor pressure data after filtering.")
    else:
        print("No CSV data.")