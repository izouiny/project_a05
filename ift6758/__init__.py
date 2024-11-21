from ift6758.features import load_advanced_train_test_dataframes

if __name__ == "__main__":
    train, test = load_advanced_train_test_dataframes()
    print(train.shape)
    print(train.head())
    print(test.shape)
    print(test.head())