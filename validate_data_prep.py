import numpy as np

# File paths
X_TRAIN_FILE = "X_train_scaled.npy"
X_TEST_FILE = "X_test_scaled.npy"
Y_TRAIN_FILE = "y_train_binary.npy"
Y_TEST_FILE = "y_test_binary.npy"

# Expectations for validation
EXPECTED_FEATURES = 190
EXPECTED_Y_VALUES = {0, 1}  # we're only working with binary classification

def validate_npy_files():
    """Utility to check shape, dtype, and contents of training/test data files."""
    print("\nStarting data file validation\n")
    
    # Load the arrays
    try:
        x_train = np.load(X_TRAIN_FILE)
        x_test = np.load(X_TEST_FILE)
        y_train = np.load(Y_TRAIN_FILE)
        y_test = np.load(Y_TEST_FILE)
    except FileNotFoundError as err:
        print(f"Critical: File missing: {err}. Did you run the data prep step?")
        return
    
    # Check shapes and data types
    print("Checking array shapes and data types...\n")
    for label, arr in [('X_train', x_train), ('X_test', x_test)]:
        if arr.ndim != 2:
            print(f"{label}: Expected 2D array, got {arr.ndim}D.")
        
        if arr.shape[1] != EXPECTED_FEATURES:
            print(f"{label}: Feature mismatch. Expected {EXPECTED_FEATURES}, got {arr.shape[1]}")
        
        if not np.issubdtype(arr.dtype, np.floating):
            print(f"{label}: Unexpected dtype {arr.dtype}. Should be float.")
        
        print(f"{label} — shape: {arr.shape}, dtype: {arr.dtype}")

    for label, arr in [('y_train', y_train), ('y_test', y_test)]:
        correct_shape = arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)
        if not correct_shape:
            print(f"{label}: Unexpected shape {arr.shape}. Expected 1D or (N, 1).")
        
        if not np.issubdtype(arr.dtype, np.integer):
            print(f"{label}: Labels should be integers. Got {arr.dtype}.")

        print(f"{label} — shape: {arr.shape}, dtype: {arr.dtype}")

    # Check scaling and label range
    print("\nReviewing feature scaling and label values...\n")

    print(f"X_train range: min={x_train.min():.4f}, max={x_train.max():.4f}")
    print(f"X_test  range: min={x_test.min():.4f}, max={x_test.max():.4f}")
    
    def check_range(array, name):
        arr_min, arr_max = array.min(), array.max()
        is_valid_range = arr_min >= -1e-6 and arr_max <= 1 + 1e-6
        collapsed = np.isclose(arr_min, arr_max)
        
        if not is_valid_range:
            print(f"{name} is outside expected [0, 1] range: min={arr_min:.6f}, max={arr_max:.6f}")
        elif collapsed:
            print(f"{name} appears to have a collapsed range (min ≈ max). Consider reviewing the scaling step.")
        else:
            print(f"{name} scaling is within expected bounds.")

    check_range(x_train, "X_train")
    check_range(x_test, "X_test")

    # Validate label values
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)

    if set(unique_train) != EXPECTED_Y_VALUES or set(unique_test) != EXPECTED_Y_VALUES:
        print("Label values are unexpected:")
        print(f"   y_train: {unique_train}")
        print(f"   y_test:  {unique_test}")
    else:
        print("Labels look correct (binary: 0 and 1)")

    # Check class distribution
    print("\nTraining set class distribution:\n")
    try:
        counts = np.bincount(y_train)
        n_zeros = counts[0] if len(counts) > 0 else 0
        n_ones = counts[1] if len(counts) > 1 else 0
        print(f"   Normal (0): {n_zeros}")
        print(f"   Attack (1): {n_ones}")
        if n_ones == 0:
            print("No attack samples found. Imbalance ratio cannot be calculated.")
        else:
            print(f"   Imbalance ratio (normal/attack): {n_zeros / n_ones:.2f}")
    except Exception as e:
        print(f"Error while computing label distribution: {e}")

if __name__ == "__main__":
    validate_npy_files()
