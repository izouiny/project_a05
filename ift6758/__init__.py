from ift6758.features import load_advanced_train_test_dataframes, load_advanced_dataframe, get_preprocessing_pipeline

preprocess = get_preprocessing_pipeline()

if __name__ == "__main__":
    # train, test = load_advanced_train_test_dataframes()
    # print(train.shape)
    # print(train.head())
    # print(test.shape)
    # print(test.head())

    # Apply data transformations
    data = load_advanced_dataframe(2018)
    train_transformed = preprocess.fit_transform(data)
    print(train_transformed.shape)
