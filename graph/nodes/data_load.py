#new file for loading data *22/01/2026

import pandas as pd
from sklearn.model_selection import train_test_split

# Phishing = 0, Benign = 1
df_urls_only = pd.read_csv("data/only_urls.csv")
df_labeled_urls = pd.read_csv("data/new_data_urls.csv")


# ---- CONFIGURABLE SPLIT RATIOS ----
dev_size = 0.50
val_size = 0.25
test_size = 0.25

assert abs(dev_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1"

# Enforce label integrity early
df_labeled_urls["status"] = df_labeled_urls["status"].astype(int)
assert set(df_labeled_urls["status"].unique()).issubset({0, 1})

# ---- STRATIFIED SPLITS ----
df_dev, df_temp = train_test_split(
    df_labeled_urls,
    test_size=(1 - dev_size),
    stratify=df_labeled_urls["status"],
    random_state=42,
)

df_val, df_test = train_test_split(
    df_temp,
    test_size=(test_size / (val_size + test_size)),
    stratify=df_temp["status"],
    random_state=42,
)

# ---- CLASS BALANCE REPORT ----
def class_balance(df, name):
    print(f"\n{name} class balance:")
    print(df["status"].value_counts(normalize=True).round(3))


class_balance(df_labeled_urls, "Full dataset")
class_balance(df_dev, "Dev split")
class_balance(df_val, "Validation split")
class_balance(df_test, "Test split")
