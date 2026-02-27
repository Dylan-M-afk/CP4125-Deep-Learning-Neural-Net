import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1) Load dataset
df = pd.read_csv("synthetic_health_risk.csv")
X = df.drop("risk", axis=1)
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3) Build a simple neural network
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    # TODO: CHANGE ONE setting below:
    # - change neurons (e.g., 8 -> 32), OR
    # - change activation (e.g., "relu" -> "tanh")
    Dense(8, activation="tanh"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=50, verbose=0)

print(
    "Test Accuracy:",
    model.evaluate(X_test, y_test, verbose=0)[1]
)

# 4) SHAP (global + local)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# Local explanation (pick one patient)
idx = 0
shap.plots.waterfall(shap_values[idx])

# 5) LIME (local explanation for the SAME patient)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=X.columns,
    class_names=["Low Risk", "High Risk"],
    mode="classification"
)
def predict(X):
    prob = model.predict(X)
    return np.hstack([1 - prob, prob])

lime_exp = lime_explainer.explain_instance(
    X_test[idx],
    predict,
    num_features=6
)

# If you run in Jupyter:
# lime_exp.show_in_notebook()

# If you run in a normal .py script, print as text:
print(lime_exp.as_list())