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

def train_xgboost(df):

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

    vp_skew = skew(np.log(df['VapourPressure_kPa']),  nan_policy='omit')
    vp_kurt = kurtosis(np.log(df['VapourPressure_kPa']), fisher=False,  nan_policy='omit')
    print("\n")
    print(f"Skewness: {vp_skew:.3f}, Kurtosis: {vp_kurt:.3f}")

    #kurtosis 2.05
    #skewness -0.58

    #feature correlation
    plt.rcParams['savefig.dpi'] = 600
    corr = df[features + ['VapourPressure_kPa']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    #feature weightage
    importance_dict = model.get_booster().get_score(importance_type='weight')
    feat_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': [importance_dict.get(f'f{i}', 0) for i in range(len(features))]
    })

    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
    plt.title("XGBoost Feature Importance (by Weight)")
    plt.xlabel("Importance Score (Split Frequency)")
    plt.ylabel("Feature (Molecular Descriptor)")
    plt.tight_layout()
    plt.show()

    #freq histogram
    sns.histplot(np.log(df['VapourPressure_kPa']), bins=30, kde=True)
    plt.title(f"VP Dist (Skewness = {vp_skew:.2f}, Kurtosis = {vp_kurt:.2f})")
    plt.xlabel("Vapour Pressure (kPa)")
    plt.ylabel("Frequency")
    plt.show()

    #exp vs pred graph
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
    csv_file = "vp_0-100.csv"
    df = load_csv_data(csv_file)

    if not df.empty:
        df = df[df['InChI'].parallel_apply(is_alkaneetc)]

        print(f"Filtered {len(df)} cleaned hydrocarbon records with vapor pressure data.")

        if not df.empty:
            trained_model = train_xgboost(df)
        else:
            print("No vapor pressure data after filtering.")
    else:
        print("No CSV data.")