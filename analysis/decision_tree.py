"""
STEP 3: Which room label best splits rooms by daytime energy per square foot?

A regression decision tree tries every label and every cut point, and keeps the split that makes the
groups' energy values least spread out (lowest variance). It repeats that inside each group.
Runs once per energy method (ILM, NILM) and per building.

Put this file in:  <repo>/analysis/decision_tree.py

In:   output/analysis/profiles_<METHOD>.csv   (from step 2)
Out:  output/analysis/tree_<METHOD>_<building>.png          picture of the tree
      output/analysis/tree_leaves_<METHOD>_<building>.csv   the final groups
      output/analysis/tree_summary.csv                      one row per method and building

Run (from the repo main folder):  python3 analysis/decision_tree.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save pictures to files without opening windows
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import cross_val_score, KFold
from config import PROCESSED, OUTPUTS, METHODS

# ---- settings you can change ------------------------------------------------
BUILDINGS = ["DBH", "SBSG"]   # each building gets its own tree (the buildings run too differently to mix)
MAX_DEPTH = 3                 # how many levels of questions
MIN_LEAF = 8                  # smallest group the tree is allowed to make
USE_LOG = True                # model log10(kBTU/sqft/day) so a few very high rooms don't dominate
N_SHUFFLES = 200              # for the "is it better than luck?" test
N_RESAMPLES = 200             # for the "is the first split stable?" test
SIZE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "M1", "M2", "M3", "M4", "L1", "XL1"]
# -----------------------------------------------------------------------------

rng = np.random.default_rng(0)


def new_tree():
    """A fresh, unfitted tree with our settings."""
    return DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=0)


def make_inputs(d):
    """Turn the labels into the numeric table X that scikit-learn needs."""
    X = pd.DataFrame(index=d.index)
    X["sq_ft"] = d["sq_ft"]                                                    # a number, as is
    X["size_bucket"] = d["size_fine"].map({b: i + 1 for i, b in enumerate(SIZE_ORDER)})  # S1=1 ... XL1=12
    X["floor"] = d["floor"]                                                    # a number, as is
    X = X.join(pd.get_dummies(d["ahu"], prefix="ahu").astype(int))             # one 0/1 column per air handler
    X = X.join(pd.get_dummies(d["room_type"], prefix="type").astype(int))      # one 0/1 column per room type
    return X


def analyze(method, building, d):
    X = make_inputs(d)
    y = np.log10(d["kbtu_per_sqft_day"]) if USE_LOG else d["kbtu_per_sqft_day"]
    print(f"\n{'=' * 70}\n{method} / {building}: {len(d)} profiles, {X.shape[1]} input columns")
    print("Input columns:", list(X.columns))

    # 1. Fit the tree on all rooms -------------------------------------------------
    tree = new_tree().fit(X, y)
    print(f"\nThe tree (values are {'log10 ' if USE_LOG else ''}kBTU per sq ft per day):")
    print(export_text(tree, feature_names=list(X.columns), decimals=2))

    # 2. Which labels mattered? -------------------------------------------------------
    imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature importance (share of the total variance reduction):")
    print(imp[imp > 0].round(3).to_string())

    # 3. Is it real? R^2 on rooms the tree did NOT see, compared with shuffled energy ----------
    #    R^2: 0 = no better than guessing the average for every room; 1 = perfect.
    cv = KFold(5, shuffle=True, random_state=0)
    real = cross_val_score(new_tree(), X, y, cv=cv, scoring="r2").mean()
    shuffled = np.array([cross_val_score(new_tree(), X, rng.permutation(y.values), cv=cv, scoring="r2").mean()
                         for _ in range(N_SHUFFLES)])
    p = (np.sum(shuffled >= real) + 1) / (N_SHUFFLES + 1)
    print(f"\nR^2 on unseen rooms (5-fold cross-validation): {real:.3f}")
    print(f"Same with shuffled energy values: mean {shuffled.mean():.3f}, best {shuffled.max():.3f}  ->  p = {p:.3f}")
    print(f"R^2 on the rooms it trained on (always higher): {tree.score(X, y):.3f}")

    # 4. Is the first split stable? Refit on resampled rooms -------------------------------
    firsts = []
    for _ in range(N_RESAMPLES):
        i = rng.integers(0, len(d), len(d))                      # draw rooms with replacement
        t = new_tree().fit(X.iloc[i], y.iloc[i])
        firsts.append(X.columns[t.tree_.feature[0]])
    first_share = pd.Series(firsts).value_counts(normalize=True)
    print(f"\nFirst split when the rooms are resampled ({N_RESAMPLES} refits):")
    print(first_share.round(2).head(5).to_string())

    # 5. The final groups (leaves): within-group spread vs all rooms -------------------------
    d = d.copy()
    d["leaf"] = tree.apply(X)
    logv = np.log10(d["kbtu_per_sqft_day"])
    leaves = d.groupby("leaf").agg(
        n=("profile_id", "size"),
        median_kbtu_per_sqft_day=("kbtu_per_sqft_day", "median"),
        spread_log10_std=("kbtu_per_sqft_day", lambda v: np.log10(v).std()),
        main_type=("room_type", lambda v: v.value_counts().index[0]),
        types=("room_type", lambda v: dict(v.value_counts())),
    ).sort_values("median_kbtu_per_sqft_day")
    within = np.mean([logv[d["leaf"] == lf].var() for lf in d["leaf"].unique()])
    print("\nFinal groups (lowest to highest energy):")
    print(leaves.round(3).to_string())
    print(f"Spread inside groups vs all rooms (variance of log10): {within:.3f} vs {logv.var():.3f}")

    leaves.to_csv(OUTPUTS / f"tree_leaves_{method}_{building}.csv")
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(tree, feature_names=list(X.columns), filled=True, impurity=False, precision=2, fontsize=9, ax=ax)
    ax.set_title(f"{method} / {building}: regression tree on daytime {'log10 ' if USE_LOG else ''}kBTU per sq ft per weekday")
    plt.tight_layout()
    plt.savefig(OUTPUTS / f"tree_{method}_{building}.png", dpi=130)
    plt.close()
    print(f"Saved tree_{method}_{building}.png and tree_leaves_{method}_{building}.csv in {OUTPUTS}")

    return {"method": method, "building": building, "profiles": len(d),
            "first_split": X.columns[tree.tree_.feature[0]], "first_split_stable_share": round(first_share.iloc[0], 2),
            "top_importances": "; ".join(f"{k} {v:.2f}" for k, v in imp[imp > 0].head(3).items()),
            "cv_r2": round(real, 3), "shuffled_r2_mean": round(shuffled.mean(), 3), "p_value": round(p, 3),
            "train_r2": round(tree.score(X, y), 3)}


summary = []
for method in METHODS:
    profiles = pd.read_csv(PROCESSED / f"profiles_{method}.csv")
    for building in BUILDINGS:
        d = profiles[profiles["building"] == building].reset_index(drop=True)
        summary.append(analyze(method, building, d))

summary = pd.DataFrame(summary)
summary.to_csv(OUTPUTS / "tree_summary.csv", index=False)
print(f"\n{'=' * 70}\nSUMMARY (also saved as tree_summary.csv)")
print(summary.to_string(index=False))

# Check yourself:
#   - In each printed tree, find the first split. Does the feature importance agree it is the most important label?
#   - Why is the cross-validated R^2 lower than the R^2 on the training rooms?
#   - If p is above 0.05, what should you conclude about that tree?
#   - Do ILM and NILM pick the same first split? If not, which would you trust, and why?