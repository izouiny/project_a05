from sklearn.base import TransformerMixin


class ColumnDropperTransformer(TransformerMixin):

    def __init__(self,columns):
        self.columns=columns

    def transform(self, X ,y=None):
        if len(self.columns) == 0:
            return X
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self
