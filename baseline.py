import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import cross_validate, train_test_split, KFold
from collections import Counter, defaultdict
import numpy as np
import random

np.random.seed(10701)
random.seed(10701)


# Read Ratings Data
ratings_path = "ratings.dat"

ratings_df = pd.read_csv(
    ratings_path,
    sep="::",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
    engine="python"
)

print(ratings_df.head())
print(ratings_df.shape)


# Read Movies Data
movies_path = "movies.dat"

movies_df = pd.read_csv(
    movies_path,
    sep="::",
    names=["MovieID", "Title", "Genres"],
    engine="python",
    encoding="latin-1"
)

print(movies_df.head())
print(movies_df.shape)


# Helper functions for MAE and RMSE

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def eval(model, ratings_df, model_name, cv=5,random_state=10701, 
            cold_max_count=4, heavy_min_count=101, contr_min_std=1.5):
    
    reader = Reader(rating_scale=(1, 5))
    kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
    # item_summary = (ratings_df.groupby('MovieID')['Rating'].agg(count='size', std='std'))
    item_summary = ratings_df.groupby('MovieID').apply(
        lambda df: pd.Series({
            'count': len(df),
            'std': np.std(df['Rating'].to_numpy(dtype=float))
        })
    )

    cold_items = set(item_summary.index[item_summary['count'] <= cold_max_count])
    heavy_items = set(item_summary.index[item_summary['count'] >= heavy_min_count])
    contr_items = set(item_summary.index[item_summary['std'] > contr_min_std])

    results = {}

    def subset_eval(name, df):
        if df.empty:
            results[name] = None
            return
        data = Dataset.load_from_df(df[["UserID", "MovieID", "Rating"]], reader)
        cv_res = cross_validate(model, data, cv=kf, measures=['RMSE', 'MAE'], verbose=False)
        results[name] = {
            'mae': np.mean(cv_res['test_mae']),
            'rmse': np.mean(cv_res['test_rmse']),
            'count': len(df)
        }

    subset_eval('all', ratings_df)
    subset_eval('cold',  ratings_df[ratings_df['MovieID'].isin(cold_items)])
    subset_eval('heavy', ratings_df[ratings_df['MovieID'].isin(heavy_items)])
    subset_eval('controversial', ratings_df[ratings_df['MovieID'].isin(contr_items)])

    print(f"Results for {model_name}")

    for subset in ['all', 'cold', 'heavy', 'controversial']:
        print(f"\n{subset} Items:")
        res = results[subset]
        if res is None:
            print("No such items")
        else:
            print(f"  Count: {res['count']}")
            print(f"  MAE: {res['mae']:.4f}")
            print(f"  RMSE: {res['rmse']:.4f}")

    return results



# Calculate MAE and RMSE 

# SVD's MAE and RMSE
svd_model = SVD(
    biased=True,
    n_factors=20,
    n_epochs=50,
    lr_all=0.005,
    reg_all=0.02
)
svd_item_results = eval(svd_model, ratings_df, "SVD")



# MF's MAE and RMSE
mf_model = SVD(
    biased=False,
    n_factors=20,
    n_epochs=30,
    lr_all=0.01,
    reg_all=0.01
)
mf_item_results = eval(mf_model, ratings_df, "Matrix Factorization")



# UserKNN's MAE and RMSE
userknn_model = KNNBasic(
    k=20, 
    sim_options={
        'name': 'pearson',
        'user_based': True
    }
)
userknn_item_results = eval(userknn_model, ratings_df, "UserKNN")



# ItemKNN's MAE and RMSE
itemknn_model = KNNBasic(
    k=20, 
    sim_options={
        'name': 'pearson',
        'user_based': False
    }
)
itemknn_item_results = eval(itemknn_model, ratings_df, "ItemKNN")




# Calculate HR@10

def calculate_hr_at_k(model, ratings_df, model_name, k=10, test_size=0.2, random_state=10701):
    
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings_df[["UserID", "MovieID", "Rating"]], reader)
    trainset, testset = train_test_split(dataset, test_size=test_size, random_state=random_state)
    
    model.fit(trainset)

    user_testset = defaultdict(set)
    [user_testset[uid].add(iid) for uid, iid, _ in testset]
    anti_testset = trainset.build_anti_testset()
    preds = model.test(anti_testset)
    user_pred_scores = defaultdict(list)
    [user_pred_scores[p.uid].append((p.iid, p.est)) for p in preds]


    hits = 0
    num_evaluated = 0

    for uid in sorted(user_testset):

        preds = user_pred_scores.get(uid)
        if preds is None:
            continue
            
        num_evaluated += 1
        ground_truth_items = user_testset[uid]
        top_k = [iid for iid, _ in sorted(preds, key=lambda x: (-x[1], x[0]))[:k]]

        if ground_truth_items.intersection(top_k):
            hits += 1

    hr = hits / num_evaluated if num_evaluated > 0 else 0.0

    print(f"{model_name}: {hr:.4f}")
    return hr


print("HR@10 on All Data")

hr10_results = {}

models = {
    "SVD": svd_model,
    "MF": mf_model,
}

for model_name, model in models.items():
    hr10_results[model_name] = calculate_hr_at_k(model, ratings_df, model_name, k=10)





