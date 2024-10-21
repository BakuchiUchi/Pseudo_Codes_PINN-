import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Step 1: Simulate the dataset
# Let's assume we have a dataset of 1000 samples with 35 risk factors
np.random.seed(42)
X = np.random.rand(1000, 35)  # 1000 samples, 35 risk factors (features)
y = np.random.randint(0, 2, 1000)  # Binary target (risk: 0 - low, 1 - high)

# Convert to pandas DataFrame for easier handling
df = pd.DataFrame(X, columns=[f"RiskFactor_{i+1}" for i in range(35)])
df['Risk'] = y

# Step 2: Attribute Reduction using a simple correlation-based selection
# (This simulates the Rough Set Theory's attribute reduction)
# We'll use SelectKBest from sklearn to select the top 12 features
def attribute_reduction(X, y, num_features=12):
    selector = SelectKBest(f_classif, k=num_features)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

# Reduce the attributes to the top 12
X_reduced, selected_features = attribute_reduction(X, y, num_features=12)

# Print the selected features (simulating RST's attribute reduction)
print(f"Selected risk factors after reduction: {selected_features + 1}")  # Adding 1 to match feature indices

# Step 3: Train Backpropagation Neural Network (BPNN)
# Split the reduced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Create a neural network model (MLPClassifier is a simple BPNN)
model = MLPClassifier(hidden_layer_sizes=(12, 8, 4), max_iter=500, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the BPNN model after attribute reduction: {accuracy * 100:.2f}%")

