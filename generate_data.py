import numpy as np
import pandas as pd

def generate_dataset(filename="data.csv", n_samples=2000, n_features=400):
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate inputs: 400 standard-scaled real numbers (mean=0, std=1)
    X = np.random.randn(n_samples, n_features)

    # Define the number of informative features based on the project's findings
    n_informative_class = 40
    n_informative_reg = 80

    # Randomly select indices for the features that will actually affect the targets
    class_indices = np.random.choice(n_features, n_informative_class, replace=False)
    reg_indices = np.random.choice(n_features, n_informative_reg, replace=False)

    # --- Classification Task: 'Class' variable ---
    # Create a non-linear dependency using the selected features and a sigmoid function
    weights_class = np.random.uniform(-1, 1, n_informative_class)
    logits = X[:, class_indices] @ weights_class
    probs = 1 / (1 + np.exp(-logits))
    y_class = (probs > 0.5).astype(int)

    # --- Regression Task: 'Output' variable ---
    # Create a linear dependency using the selected features and add some Gaussian noise
    weights_reg = np.random.uniform(-2, 2, n_informative_reg)
    noise = np.random.randn(n_samples) * 2.0
    y_reg = X[:, reg_indices] @ weights_reg + noise

    # --- Build the DataFrame ---
    # Create column names exactly as expected: Input1 to Input400
    feature_names = [f"Input{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add target variables to the DataFrame
    df["Class"] = y_class
    df["Output"] = y_reg

    # Save to a CSV file using ';' as the separator to match the notebook code
    df.to_csv(filename, sep=";", index=False)
    print(f"Dataset successfully generated and saved to {filename}!")
    print(f"Dimensions: {df.shape[0]} samples, {df.shape[1]} columns.")

if __name__ == "__main__":
    generate_dataset()