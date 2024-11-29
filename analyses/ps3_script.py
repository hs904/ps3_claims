# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import dalex as dx
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import sys
import os

# Add the parent directory of `ps3` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ps3.data import create_sample_split, load_transform


# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
df.head()
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
#Claims are proportional to the duration or level of risk insured.
#The outcome variable (Pure Premium) represents the normalized claim intensity, enabling fair comparison and effective predictive modeling.

# TODO: use your create_sample_split function here
df = create_sample_split(df, "IDpol", 0.8)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
#Increase the max_iter parameter in the GeneralizedLinearRegressor:
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True,max_iter=500)
#max_iter should ensure the solver has enough iterations to converge.
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here
        ("numeric",
         Pipeline(
             [
                 ("scale", StandardScaler()),
                 ("spline",SplineTransformer(include_bias=False,knots="quantile")),
             ]
         ),
          numeric_cols,
        ),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
model_pipeline = Pipeline(
    # TODO: Define pipeline steps here
     [
         ("preprocess",preprocessor),
      (
          "estimate",
          GeneralizedLinearRegressor(
              family=TweedieDist, l1_ratio=1, fit_intercept=True
                ),
        ),   
     ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
# Define the LightGBM model pipeline
lgbm_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("estimate", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5)),
    ]
)
cv = GridSearchCV(
    lgbm_pipeline,
    param_grid={
        "estimate__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        "estimate__n_estimators": [50, 100, 150, 200],
    },
    verbose=2,
        cv=5,
)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %% Now let's get started with Problem Set 4!
###############################################
#Ex_1 Monotonicity constraints
#1. ceate a plot of the average claims per BonusMalus group, make sure to weigh them by exposure.
# Group data by BonusMalus and calculate weighted averages

weighted_avg_claims = (
    df.groupby("BonusMalus")
    .apply(lambda group: np.average(group["PurePremium"], weights=group["Exposure"]))
)

# Plot the weighted average claims
plt.figure(figsize=(10, 6))
plt.plot(weighted_avg_claims.index, weighted_avg_claims.values, marker="o")
plt.title("Weighted Average Claims per BonusMalus Group")
plt.xlabel("BonusMalus")
plt.ylabel("Weighted Average Claims")
plt.grid()
plt.show()

# What happens if we don’t include a monotonicity constraint?
# Without the constraint, complex models like GBM might overfit or introduce interactions that break this expected monotonicity. This can lead to unrealistic or counterintuitive pricing strategies.

# 2. introduce a monotonicity constraint for BonusMalus by setting a value of 1 in a list corresponding to all features (0 for others).
# Define monotonicity constraints
monotonicity_constraints = [0] * len(categoricals) + [1, 0]  

# Define the constrained LGBM pipeline
constrained_lgbm_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("estimate", LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            monotone_constraints=monotonicity_constraints
        )),
    ]
)

#3. Cross-validate constrained LGBM
constrained_cv = GridSearchCV(
    constrained_lgbm_pipeline,
    param_grid={
        "estimate__learning_rate": [0.01, 0.02, 0.05],
        "estimate__n_estimators": [100, 200],
    },
    verbose=2,
    cv=5,
)
constrained_cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

# Save predictions
df_test["pp_t_lgbm_constrained"] = constrained_cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = constrained_cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)


###############################################
#Ex_2 Learning Curve
#1. Refit the best constrained lgbm estimator from the cross-validation and provide the tuples of the test and train dataset to the estimator via eval_set . 
best_constrained_lgbm = constrained_cv.best_estimator_.named_steps["estimate"]
best_constrained_lgbm.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t,
    eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    eval_sample_weight=[w_train_t, w_test_t],
    eval_metric="rmse",
    verbose=10,
    early_stopping_rounds=20, 
)

#2. Plot the learning curve
# Plot learning curve
lgb.plot_metric(best_constrained_lgbm.evals_result_, metric="rmse")
plt.title("Learning Curve: Constrained LGBM")
plt.xlabel("Iterations")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.legend(["Training", "Validation"])
plt.grid()
plt.show()

#3. What do you notice, is the estimator tuned optimally?

###############################################
#Ex_3 Metrics Function
from evaluation._evaluate_predictions import evaluate_predictions

# Evaluate the unconstrained LGBM model
# Use the function and compare the constrained and unconstrained lgbm models.
metrics_unconstrained = evaluate_predictions(
    y_true=y_test_t,
    y_pred=df_test["pp_t_lgbm"],
    exposure=df_test["Exposure"],
    sample_weight=w_test_t,
)

# Evaluate the constrained LGBM model
metrics_constrained = evaluate_predictions(
    y_true=y_test_t,
    y_pred=df_test["pp_t_lgbm_constrained"],
    exposure=df_test["Exposure"],
    sample_weight=w_test_t,
)

# Display results
print("Unconstrained LGBM Metrics:")
print(metrics_unconstrained)

print("\nConstrained LGBM Metrics:")
print(metrics_constrained)

# %% 
# Plots the PDPs of all features and compare the PDPs between the unconstrained and constrained LGBM. 

# Step 1: Define an explainer object for the constrained LGBM model
# Pass the model, training data (X_train), and target values (y_train)
explainer_constrained = dx.Explainer(best_constrained_lgbm, X_train_t, y_train_t)

# Step 2: Compute the marginal effects (PDPs) for the constrained model
pdp_constrained = explainer_constrained.model_profile()

# Step 3: Plot PDP for the constrained model
pdp_constrained.plot()

# Step 4: Define an explainer object for the unconstrained LGBM model
# Pass the unconstrained model, training data (X_train), and target values (y_train)
best_unconstrained_lgbm=cv.best_estimator_.named_steps["estimate"]
explainer_unconstrained = dx.Explainer(best_unconstrained_lgbm, X_train_t, y_train_t)

# Step 5: Compute the marginal effects (PDPs) for the unconstrained model
pdp_unconstrained = explainer_unconstrained.model_profile()

# Step 6: Plot PDP for the unconstrained model
pdp_unconstrained.plot()

# Step 7: Optional: Compare the PDPs of both models
# Create a new plot with both models' PDPs on the same graph for comparison
fig, ax = plt.subplots(figsize=(10, 6))
pdp_constrained.plot(ax=ax, label='Constrained Model')
pdp_unconstrained.plot(ax=ax, label='Unconstrained Model')
plt.legend()
plt.show()

# %%
# Compare the decompositions of the predictions for some specific row for the constrained LGBM and our initial GLM.

# Select a specific data point (e.g., the first row of the test set)
data_point = X_test_t.iloc[[0]]

# Compute SHAP decompositions for the selected data point
shap_lgbm = explainer_constrained.predict_parts(data_point, type="shap")
shap_glm = explainer_unconstrained.predict_parts(data_point, type="shap")

# Plot the decompositions for both models
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# SHAP decomposition plot for the constrained LGBM
shap_lgbm.plot(show=False, ax=axes[0])
axes[0].set_title("SHAP Decomposition - Constrained LGBM")

# SHAP decomposition plot for the initial GLM
shap_glm.plot(show=False, ax=axes[1])
axes[1].set_title("SHAP Decomposition - Initial GLM")

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
