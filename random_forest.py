# Surrogate Model Pipeline for Vapour Pressure of Alcohols (CSV-based)

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
def load_csv_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return pd.DataFrame()

def compute_descriptors(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumBranches': num_branches(inchi)
        }
    else:
        return {
            'MolWt': np.nan,
            'TPSA': np.nan,
            'NumHDonors': np.nan,
            'NumHAcceptors': np.nan,
            'MolLogP': np.nan,
            'NumBranches': np.nan
        }

def is_alkaneetc(inchi):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ["C", "H"]:
            return False
    return True

def num_branches(inchi):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return False
    branch_points = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":  # focus on hydrocarbons
            if atom.GetDegree() > 2:  # more than 2 bonds → branch
                branch_points += 1
    return branch_points

def train_random_forest(df):

    desc_df = df['InChI'].apply(compute_descriptors).apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1).dropna()

    features = ['Temperature_K', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP', 'NumBranches']
    X = df[features]
    y = np.log(df['VapourPressure_kPa'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = RandomForestRegressor(
        n_estimators=100,      
        max_depth=None,               
        n_jobs=-1              
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred))  
    mape = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))
    r2 = r2_score(y_test, y_pred)
    plt.scatter(y_test, y_pred)
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.show()
    print("\nRandom Forest Model Evaluation:")
    print("MAE (kPa):", mae)
    print("MAPE: ", mape)
    print("R^2 Score:", r2)
    return model

if __name__ == "__main__":
    csv_file = "vapour_pressure_data_final.csv"
    df = load_csv_data(csv_file)

    if not df.empty:
        df = df[df['InChI'].apply(is_alkaneetc)]

        df = df[df['VapourPressure_kPa'] > 0.00001]
        df = df[df['VapourPressure_kPa'] < 8000]
        print(f"Filtered {len(df)} cleaned linear hydrocarbon records with vapor pressure data.")

        print("Min P (kPa):", df['VapourPressure_kPa'].min())
        print("Max P (kPa):", df['VapourPressure_kPa'].max())

        if not df.empty:
            trained_model = train_random_forest(df)
        else:
            print("No vapor pressure data after filtering.")
    else:
        print("No CSV data.")