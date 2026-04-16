import pandas as pd
from sklearn.model_selection import train_test_split

#the correct version of data loading in the ensemble branch where labels are normalized. this correlates with ensemble2.py and eval.py

#Phishing score: 0, Benign score: 1

def normalize_labels(df: pd.DataFrame, label_col: str, phishing_value: int) -> pd.DataFrame:
    """
    Converts dataset-specific labels into internal convention:
    Benign = 1
    Phishing = 0
    """
    df = df.copy()

    # phishing_value indicates which value in the dataset means "phishing"
    df[label_col] = df[label_col].astype(int)
    if phishing_value == 1:
        # Dataset uses: 1 = Phishing, 0 = Benign
        df[label_col] = df[label_col].map({1: 0, 0: 1})
    elif phishing_value == 0:
        # Dataset already matches internal convention
        pass
    else:
        raise ValueError("Unsupported phishing label encoding")

    # Final sanity check
    assert set(df[label_col].unique()).issubset({0, 1}), "Invalid labels after normalization"

    return df

#Old datasets where Phishing= 0, Benign = 1
df_urls_only = pd.read_csv('data/only_urls.csv')
df_labeled_urls_old = pd.read_csv('data/new_data_urls.csv')
df_labeled_urls = normalize_labels(df_labeled_urls_old, label_col="status", phishing_value=0)
old_sample = df_labeled_urls.sample(frac=0.0001, random_state=42)

print(f"Total OLD URLs with labels: {len(df_labeled_urls)}")
print(f"Sampled Old URLs with labels: {len(old_sample)}")
# print(f"Old Phishing: {(old_sample['status'] == 0).sum()}")
# print(f"Old Benign: {(old_sample['status'] == 1).sum()}")

df_labeled_urls["status"] = df_labeled_urls["status"].astype(int)
assert set(df_labeled_urls["status"].unique()).issubset({0, 1})

old_sample["status"] = old_sample["status"].astype(int)
assert set(old_sample["status"].unique()).issubset({0, 1})


#New dataset where Phishing = 1, Benign = 0
df_new_urls = pd.read_csv("data/phishing_url_dataset_unique.csv")
df_new_url = normalize_labels(df_new_urls, label_col="label",phishing_value=1)

phish_frac  = 0.0039   # sample less phishing
benign_frac = 0.0059  # sample more benign

df_phish = df_new_url[df_new_url["label"] == 0].sample(frac=phish_frac, random_state=42)
df_benign = df_new_url[df_new_url["label"] == 1].sample(frac=benign_frac, random_state=42)

url_sample = pd.concat([df_phish, df_benign]).sample(frac=1, random_state=42)
#url_sample = df_new_url.sample(frac=0.01, random_state=42)
print(f"Total New Labeled URLs: {len(df_new_url)}")
# print(f"Phishing: {(df_new_url['label'] == 1).sum()}")
# print(f"Benign: {(df_new_url['label'] == 0).sum()}")
print(f"Sampled New URLs with labels: {len(url_sample)}")
# print(f"Phishing in sample: {(url_sample['label'] == 0).sum()}")
# print(f"Benign in sample: {(url_sample['label'] == 1).sum()}")

# ---- CONFIGURABLE SPLIT RATIOS ----
dev_size = 0.40
val_size = 0.3
test_size = 0.3

# Enforce label integrity early
df_new_url["label"] = df_new_url["label"].astype(int)
assert set(df_new_url["label"].unique()).issubset({0, 1})

url_sample["label"] = url_sample["label"].astype(int)
assert set(url_sample["label"].unique()).issubset({0, 1})

# ---- STRATIFIED SPLITS FOR NEW DATASET ----
df_dev, df_temp = train_test_split(
    url_sample,
    test_size=(1 - dev_size),
    stratify=url_sample["label"],
    random_state=42,
)

df_val, df_test = train_test_split(
    df_temp,
    test_size=(test_size / (val_size + test_size)),
    stratify=df_temp["label"],
    random_state=42,
)

# ---- CLASS BALANCE REPORT ----
def class_balance(df, name):
    print(f"\n{name} class balance:")
    print(df["label"].value_counts(normalize=True).round(3))

class_balance(url_sample, "Full dataset - New URLs")
class_balance(df_dev, "Dev split - New URLs")
class_balance(df_val, "Validation split - New URLs")
class_balance(df_test, "Test split - New URLs")
print(f"Develop split New URLs number is: {len(df_dev)}")
print(f"Validation split New URLs number is: {len(df_val)}")
print(f"Test split New URLs number is: {len(df_test)}")


# ---- STRATIFIED SPLITS FOR OLD DATASET ----
df_dev_old, df_temp_old = train_test_split(
    old_sample,
    test_size=(1 - dev_size),
    stratify=old_sample["status"],
    random_state=42,
)

df_val_old, df_test_old = train_test_split(
    df_temp_old,
    test_size=(test_size / (val_size + test_size)),
    stratify=df_temp_old["status"],
    random_state=42,
)

# ---- CLASS BALANCE REPORT ----
def class_balance(df, name):
    print(f"\n{name} class balance:")
    print(df["status"].value_counts(normalize=True).round(3))

class_balance(old_sample, "Full dataset")
class_balance(df_dev_old, "Dev split")
class_balance(df_val_old, "Validation split")
class_balance(df_test_old, "Test split")
print(f"Dev split Old URLs number is: {len(df_dev_old)}")
print(f"Validation split Old URLs number is: {len(df_val_old)}")
print(f"Test split Old URLs number is: {len(df_test_old)}")
