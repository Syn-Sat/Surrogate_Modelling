# Surrogate Model Pipeline for Vapour Pressure of Hydrocarbons (with H,C,N atoms) (CSV-based)
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from scipy.stats import skew, kurtosis
import seaborn as sns
RDLogger.DisableLog('rdApp.warning')

pandarallel.initialize(progress_bar=True)

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
            'BalabanJ': Descriptors.BalabanJ(mol), #connectivity index that accounts for the number of edges, nodes, and distances in the molecular graph;
            'BertzCT': Descriptors.BertzCT(mol),
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
            'BalabanJ': np.nan,
            'BertzCT': np.nan,
            'NumBranches': np.nan,
            'NumNitrogens': np.nan
        }

def is_alkaneetc(inchi):
    from rdkit import Chem
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in [1, 6, 7]: #H,C,N,Cl,Br,I
            return False
    return True

def train_random_forest(df):

    desc_df = df['InChI'].parallel_apply(compute_descriptors).parallel_apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1).drop_duplicates().dropna()

    features = ['Temperature_K', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP', 'BalabanJ', 'BertzCT', 'NumBranches', 'NumNitrogens']
    X = df[features]    
    y = np.log(df['VapourPressure_kPa'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"  # use GPU if available, else remove this
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred))  
    mape = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred))
    r2 = r2_score(y_test, y_pred)

    vp_skewness = skew(df['VapourPressure_kPa'], nan_policy='omit')
    print(f"Skewness of VapourPressure_kPa: {vp_skewness:.3f}")

    # Plot histogram
    sns.histplot(df['VapourPressure_kPa'], bins=50, kde=True)
    plt.title(f"Vapour Pressure Distribution (Skewness = {vp_skewness:.2f})")
    plt.xlabel("Vapour Pressure (kPa)")
    plt.ylabel("Frequency")
    plt.show()

    plt.scatter(y_test, y_pred, color='blue')
    m, b = np.polyfit(y_test, y_pred, 1)

    plt.plot(y_test, m*y_test + b, 'r--', label=f'Fit: y={m:.2f}x+{b:.2f}')

    plt.xlabel('Experimental')
    plt.ylabel('Predicted')

    print("\nResult:")
    print("MAE (kPa):", mae)
    print("MAPE: ", mape)
    print("R^2 Score:", r2)

    plt.legend()
    plt.show()

    return model

if __name__ == "__main__":
    csv_file = "vp_0-2.csv"
    df = load_csv_data(csv_file)

    if not df.empty:
        df = df[df['InChI'].parallel_apply(is_alkaneetc)]

        print(f"Filtered {len(df)} cleaned hydrocarbon records with vapor pressure data.")

        if not df.empty:
            trained_model = train_random_forest(df)
        else:
            print("No vapor pressure data after filtering.")
    else:
        print("No CSV data.")