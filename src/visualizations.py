import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

"""
Figure 1:
Plots the distribution of transfer fees (excluding free transfers) and the
log-transformed distribution. The left subplot shows the raw fee distribution
with heavy right skew, while the right subplot shows the log-transformed version
to visualize lower-valued transfers more clearly.
"""
transfer_data = pd.read_csv('data/transfer_data.csv')
transfer_data = transfer_data[transfer_data['transfer_fee'] > 0]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(transfer_data['transfer_fee'], bins=100, alpha=0.5, label='Original', color='blue')
axes[0].set_title('Distribution of Transfer Fees')
axes[0].set_xlabel('Transfer Fee')
axes[0].set_ylabel('Frequency')
axes[0].legend()
log_fees = np.log1p(transfer_data['transfer_fee'])
axes[1].hist(log_fees, bins=100, alpha=0.5, label='Log-Transformed', color='blue')
axes[1].set_title('Distribution of Transfer Fees (log)')
axes[1].set_xlabel('Log Transfer Fee')
axes[1].set_ylabel('Frequency')
axes[1].legend()
plt.tight_layout()
plt.savefig('results/figures/figure1.png')
print(f'Figure 1 saved.')

"""
Figure 2:
Classifies transfers into five quantile-based fee classes and visualizes three aspects:
(1) The number of transfers per class,
(2) The mean transfer fee within each class,
(3) A boxplot of transfer fee distributions across classes.
This highlights how transfer fee levels vary across derived quantile groups.
"""
transfer_data['fee_class'] = pd.qcut(transfer_data['transfer_fee'], q=5, labels=[i for i in range(5)])
counts = transfer_data['fee_class'].value_counts().sort_index()
mean_values = transfer_data.groupby('fee_class', observed=True)['transfer_fee'].mean()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes[0].bar(counts.index, counts.values)
axes[0].set_title('Count of Transfers per Class')
axes[0].set_ylabel('Count')
axes[1].bar(mean_values.index, mean_values.values)
axes[1].set_title('Mean Transfer Fee per Class')
axes[1].set_ylabel('Mean Fee')
transfer_data.boxplot(
    column='transfer_fee',
    by='fee_class',
    ax=axes[2]
)
axes[2].set_title('Transfer Fee Distribution by Class')
axes[2].set_xlabel('Class')
axes[2].set_ylabel('Transfer Fee')
plt.suptitle('')
plt.tight_layout()
plt.savefig('results/figures/figure2.png')
print(f'Figure 2 saved.')

"""
Figure 3a:
Plots the median transfer fee for each selling league in the dataset.
This highlights which leagues tend to sell players at higher prices on average.
"""
mean_fee_by_league = transfer_data.groupby('selling_league')['transfer_fee'].median().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
mean_fee_by_league.plot(kind='bar')
plt.title('Median Transfer Fee by Selling League')
plt.ylabel('Median Transfer Fee (€)')
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('results/figures/figure3a.png')
print(f'Figure 3a saved.')

"""
Figure 3b:
Visualizes the relationship between a player's age at the time of transfer
and the corresponding transfer fee using a scatter plot.
"""
plt.figure(figsize=(10, 6))
plt.scatter(transfer_data['age_at_transfer'], transfer_data['transfer_fee'], alpha=0.5)
plt.title('Transfer Fee vs Age at Transfer')
plt.ylabel('Transfer Fee')
plt.xticks(rotation=45)
plt.savefig('results/figures/figure3b.png')
print(f'Figure 3b saved.')

"""
Figure 4a:
Plots train and validation R² across boosting rounds for multiple XGBoost
regularisation settings (alpha, lambda). Shows how different penalties affect
model performance and overfitting behaviour.
"""

train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")
X_train = train.drop(columns=['transfer_fee'])
X_test  = test.drop(columns=['transfer_fee'])
y_train = train['transfer_fee']
y_test  = test['transfer_fee']

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_test,  label=y_test)

regularizations = [
    (1, 5),
    (0, 5), 
    (1.0, 10.0),
]

plt.figure(figsize=(5, 5))

for alpha_val, lambda_val in regularizations:
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'alpha': alpha_val,
        'lambda': lambda_val,
        'max_depth': 6,
        'eta': 0.1
    }

    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    history = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        evals_result=history,
        verbose_eval=False
    )
    
    train_r2 = []
    val_r2 = []
    
    for i in range(len(history['train']['rmse'])):
        y_pred_train = bst.predict(dtrain, iteration_range=(0, i+1))
        y_pred_val   = bst.predict(dval,   iteration_range=(0, i+1))
        
        train_r2.append(r2_score(y_train, y_pred_train))
        val_r2.append(r2_score(y_test,  y_pred_val))
    
    plt.plot(train_r2, label=f"train R² α={alpha_val}, λ={lambda_val}")
    plt.plot(val_r2,   label=f"val R² α={alpha_val}, λ={lambda_val}")

plt.title("R² with different regularisation in XGBoost")
plt.xlabel("Boosting round")
plt.ylabel("R² score")
plt.legend()
plt.grid(True)
plt.savefig("results/figures/figure4a.png")
print(f'Figure 4 saved.')



"""
Figure 4b:
Reads CatBoost feature importances from CSV and plots the 10 most influential
regression features in a horizontal bar chart.
"""

importances = pd.read_csv(
    "results/catboost_importances.csv",
    index_col=0, header=None,
    names=["feature", "importance"],
    skiprows=1
)
top_features = importances.sort_values(by="importance", ascending=False).head(10)

plt.figure(figsize=(4, 3))
plt.barh(top_features.index, top_features['importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top Features")
plt.tight_layout()
plt.savefig("results/figures/figure4b.png")
print(f'Figure 4b saved.')

"""
Figure 4c:
Reads XGBoost classifier feature importances and plots the top 10 categorical
predictors in a horizontal bar chart to show which variables contribute most
to classification performance.
"""

importances = pd.read_csv(
    "results/xgb_classifier_importances_label.csv",
    index_col=0, header=None,
    names=["feature", "importance"]
)
top_features = importances.sort_values(by="importance", ascending=False).head(10)


plt.figure(figsize=(4, 3))
plt.barh(top_features.index, top_features['importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top Features")
plt.tight_layout()
plt.savefig("results/figures/figure4c.png")
print(f'Figure 4c saved.')


"""
Extra Figure:
Plots histograms for every numeric player metric in the dataset.
Each subplot includes the number of unique values for context.
This provides an overview of variable distributions.
"""
player_data = pd.read_csv('data/player_data.csv')
num_cols = [c for c in player_data.columns.tolist() if pd.api.types.is_numeric_dtype(player_data[c])]
n = len(num_cols)
rows = int(np.ceil(n / 5))  

fig, axes = plt.subplots(rows, 5, figsize=(20, rows * 3))
axes = axes.flatten()
for ax, col in zip(axes, num_cols):
    cleaned = player_data[col]
    unique_count = cleaned.nunique()
    ax.hist(cleaned, bins=30)
    ax.set_title(f"{col}  (unique: {unique_count})", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
for i in range(len(num_cols), len(axes)):
    axes[i].axis("off")
plt.tight_layout()
plt.savefig('results/figures/all_metrics.png')
print(f'All player metrics distributions saved.')
