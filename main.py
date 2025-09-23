# Surrogate Model Pipeline for Vapour Pressure of Alcohols (CSV-based, InChI support)

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Step 1: Load CSV File ---
def load_csv_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return pd.DataFrame()

# --- Step 2: Compute Molecular Descriptors from InChI ---
def compute_descriptors(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'MolLogP': Descriptors.MolLogP(mol)
        }
    else:
        return {
            'MolWt': np.nan,
            'TPSA': np.nan,
            'NumHDonors': np.nan,
            'NumHAcceptors': np.nan,
            'MolLogP': np.nan
        }

# --- Step 3: Filter Alcohol-like Molecules (C–O bonds) ---
def is_alkaneetc(inchi):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ["C", "H"]:
            return False
    return True

# --- Step 4: Build and Evaluate Linear Regression Model ---
def train_linear_model(df):
    # Compute descriptors
    desc_df = df['InChI'].apply(compute_descriptors).apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1).dropna()

    # Keep only positive pressures
    df = df[df['VapourPressure_kPa'] > 0]

    features = ['Temperature_K', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP']
    X = df[features]
    y = np.log(df['VapourPressure_kPa'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred))  # back-transform
    r2 = r2_score(y_test, y_pred)

    print("\nLinear Regression Model Evaluation:")
    print("MAE (kPa):", mae)
    print("R² Score:", r2)
    return model

# --- Main Execution ---
if __name__ == "__main__":
    csv_file = "thermoml_vapor_pressure.csv"  # generated from XML parser
    df = load_csv_data(csv_file)

    if not df.empty:
        df = df[df['InChI'].apply(is_alkaneetc)]
        print(f"Filtered {len(df)} alcohol-like records with vapor pressure data.")
        print(df[['InChI', 'VapourPressure_kPa', 'Temperature_K']].head(10))

        if not df.empty:
            trained_model = train_linear_model(df)
        else:
            print("No vapor pressure data after filtering.")
    else:
        print("No CSV data.")