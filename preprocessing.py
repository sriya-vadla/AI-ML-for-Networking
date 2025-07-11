import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import zipfile
import os

def load_and_preprocess_data(arff_file_path="/home/sriya/Downloads/ai_network/network_analysis/KDDTest+.arff", test_size=0.2, random_state=42):
    """
    Loads data from an ARFF file, preprocesses it, and splits it into training and testing sets.

    Args:
        arff_file_path (str): The path to the input ARFF file.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training labels.
            - y_test (pd.Series): Testing labels.
            - preprocessor (ColumnTransformer): The fitted preprocessor object.
            - df (pd.DataFrame): The loaded and initially processed DataFrame.
    """
    attributes = []
    data_lines = []
    data_section = False

    try:
        with open(arff_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith('@attribute'):
                    parts = line.split()
                    attribute_name = parts[1]
                    attributes.append(attribute_name)
                elif line.lower().startswith('@data'):
                    data_section = True
                elif data_section and line and not line.startswith('%'):
                    data_lines.append(line.split(','))
    except FileNotFoundError:
        print(f"Error: File not found at {arff_file_path}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error reading ARFF file: {e}")
        return None, None, None, None, None, None

    if not attributes or not data_lines:
        print("Error: Could not extract attributes or data from the ARFF file.")
        return None, None, None, None, None, None

    df = pd.DataFrame(data_lines, columns=attributes)
    print("✅ ARFF data loaded successfully into DataFrame!")
    print("📄 Shape of data (rows, columns):", df.shape)
    print("📌 Column names:", df.columns.tolist())

    # Separate features (X) and label (y)
    if "'class'" not in df.columns:
        print("Error: Target column ''class'' not found in DataFrame.")
        return None, None, None, None, None, None

    X = df.drop(columns=["'class'"])
    y = df["'class'"]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'object']).columns # Include object for potential conversion

    # Attempt to convert numerical columns to numeric, coercing errors
    for col in numerical_features:
         df[col] = pd.to_numeric(df[col], errors='coerce')

    # Re-identify numerical and categorical features after conversion attempt
    X = df.drop(columns=["'class'"])
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns


    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep any columns not specified (e.g., if a column type was missed)
    )

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)


    print("✅ Data preprocessing complete.")

    return X_train, X_test, y_train, y_test, preprocessor, df

def preprocess_privacy_data(privacy_df, test_size=0.2, random_state=42):
    """
    Preprocesses the simulated privacy data and splits it into training and testing sets.

    Args:
        privacy_df (pd.DataFrame): The input DataFrame with simulated privacy data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: A tuple containing:
            - Xp_train_scaled (np.ndarray): Scaled training features.
            - Xp_test_scaled (np.ndarray): Scaled testing features.
            - yp_train (pd.Series): Training labels.
            - yp_test (pd.Series): Testing labels.
            - scaler (StandardScaler): The fitted StandardScaler object.
    """
    if 'Label' not in privacy_df.columns:
        print("Error: Target column 'Label' not found in privacy DataFrame.")
        return None, None, None, None, None

    X_priv = privacy_df.drop(columns=['Label'])
    y_priv = privacy_df['Label']

    # Train-test split
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_priv, y_priv, test_size=test_size, stratify=y_priv, random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    Xp_train_scaled = scaler.fit_transform(Xp_train)
    Xp_test_scaled = scaler.transform(Xp_test)

    print("✅ Privacy data preprocessing complete.")

    return Xp_train_scaled, Xp_test_scaled, yp_train, yp_test, scaler
