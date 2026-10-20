"""Baseline (PCA reconstruction-error curation + default RBF SVM) vs.
class-balanced, CV-tuned RBF SVM on the full noisy credit-g training set.
Runs on CPU in seconds; all numbers are computed at run time."""
import io, json, sys, time
import numpy as np, pandas as pd, requests, matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
NOISE_SEEDS = [42, 43, 44]          # label-noise draws; models are deterministic given data
META_URL = "https://www.openml.org/api/v1/json/data/31"  # OpenML credit-g (data_id 31)
SOURCE_URL = "https://www.openml.org/d/31"


def get(url, tries=5):
    """HTTP GET with timeout=60 and exponential backoff on 429/5xx and network errors."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"  download failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def load_credit_g():
    """Download credit-g ARFF from OpenML and return one-hot features and labels (1 = bad credit)."""
    print("Downloading credit-g from OpenML...")
    arff_url = get(META_URL).json()["data_set_description"]["url"]
    data, _ = arff.loadarff(io.StringIO(get(arff_url).text))
    df = pd.DataFrame(data)
    for c in df.columns:  # decode byte strings from nominal attributes
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    y = (df["class"] == "bad").astype(int).to_numpy()
    X = pd.get_dummies(df.drop(columns="class"), dtype=float)  # one-hot categoricals
    return X.to_numpy(float), y


def flip_labels(y, rng, rate=0.1):
    """Flip a random 10% of training labels (label-noise simulation)."""
    y2 = y.copy()
    idx = rng.choice(len(y), int(round(rate * len(y))), replace=False)
    y2[idx] = 1 - y2[idx]
    return y2


def pca_curate(X, y, drop=0.2):
    """Baseline curation: within each (noisy) class, fit PCA and drop the 20% of points
    with the largest squared reconstruction error. Stand-in for L1-PCA curation."""
    keep = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        pca = PCA(n_components=0.9, svd_solver="full").fit(X[idx])  # keep 90% variance
        err = ((X[idx] - pca.inverse_transform(pca.transform(X[idx]))) ** 2).sum(1)
        n_keep = len(idx) - int(round(drop * len(idx)))
        keep[idx[np.argsort(err)[:n_keep]]] = True
    return keep


def evaluate(model, X, y):
    """Shared evaluation: macro-F1 (main), plus accuracy and ROC-AUC on the clean test set."""
    pred = model.predict(X)
    return {"f1": f1_score(y, pred, average="macro"),
            "acc": accuracy_score(y, pred),
            "auc": roc_auc_score(y, model.decision_function(X))}


def main():
    X, y = load_credit_g()
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features, bad-credit share {y.mean():.2f}")
    # Split ONCE, stratified, fixed seed. Test labels stay clean.
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(Xtr_raw)  # fit scaler on training data only
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
    print(f"Train {len(ytr)}, test {len(yte)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1.0]}
    runs = {"baseline": [], "improved": []}
    for s in NOISE_SEEDS:
        print(f"-- label-noise seed {s}")
        ytr_noisy = flip_labels(ytr, np.random.default_rng(s))  # same noisy labels for both arms

        # BASELINE: PCA-reconstruction curation + default RBF SVM (C=1, gamma='scale')
        keep = pca_curate(Xtr, ytr_noisy)
        base = SVC(kernel="rbf").fit(Xtr[keep], ytr_noisy[keep])
        print(f"  baseline kept {keep.sum()}/{len(keep)} training points")
        runs["baseline"].append(evaluate(base, Xte, yte))

        # IMPROVED: class-balanced RBF SVM on full noisy set, CV grid search (training labels only)
        search = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), grid, cv=cv,
                              scoring="f1_macro", n_jobs=-1).fit(Xtr, ytr_noisy)
        print(f"  improved best params {search.best_params_}")
        runs["improved"].append(evaluate(search.best_estimator_, Xte, yte))

    m = {}
    for arm in runs:
        f1s = np.array([r["f1"] for r in runs[arm]])
        m[f"{arm}_score"] = float(f1s.mean())
        m[f"{arm}_score_std"] = float(f1s.std())
        m[f"{arm}_accuracy"] = float(np.mean([r["acc"] for r in runs[arm]]))
        m[f"{arm}_roc_auc"] = float(np.mean([r["auc"] for r in runs[arm]]))
    m["improvement"] = m["improved_score"] - m["baseline_score"]  # positive = improved better
    if not all(np.isfinite(v) for v in m.values()):
        print("Non-finite metrics; failing.")
        sys.exit(1)

    # Optional chart
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline (PCA curation)", "improved (balanced SVM)"],
           [m["baseline_score"], m["improved_score"]],
           yerr=[m["baseline_score_std"], m["improved_score_std"]], capsize=6,
           color=["grey", "steelblue"])
    ax.set_ylabel("macro-F1 on clean test set (mean ± std over 3 noise seeds)")
    fig.tight_layout()
    fig.savefig("figures/macro_f1_bar.png", dpi=120)

    results = {
        "question": "Does a class-balanced, CV-tuned RBF SVM on the full noisy, imbalanced training set "
                    "beat an SVM trained on a PCA-curated subset of that set, on the public credit-g dataset?",
        "method": "credit-g (OpenML 31) one-hot + standardised; stratified 75/25 split (seed 42); 10% training "
                  "labels flipped (noise seeds 42-44); baseline drops 20% per class by PCA reconstruction error "
                  "then default RBF SVM; improved = class_weight='balanced' RBF SVM with 5-fold stratified grid "
                  "search on C and gamma; metric macro-F1 on clean test split.",
        "dataset": {"name": "credit-g (OpenML data_id 31)", "source_url": SOURCE_URL,
                    "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "macro-F1 (clean test set)",
        "higher_is_better": True,
        "baseline": "Within-class PCA reconstruction-error curation (drop 20% largest-error points per class, "
                    "90% variance PCA) followed by an RBF SVM with default C=1 and gamma='scale'.",
        "improved": "RBF SVM on the full noisy training set with class_weight='balanced' and C, gamma chosen "
                    "by 5-fold stratified grid search on noisy training labels only.",
        "metrics": m,
        "notes": "Baseline is a stand-in for the paper's L1-PCA curation (L2-PCA proxy); not the paper's method "
                 "or dataset. Only one stratified split (seed 42); variability reflects label-noise draws "
                 "(seeds 42-44) only, std is population std over 3 runs. Neither model is randomised at fit time. "
                 "Test set used once for final scoring. Class for PCA curation uses noisy labels, as in practice. "
                 "Small test set (250 rows) means noisy macro-F1 estimates.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
