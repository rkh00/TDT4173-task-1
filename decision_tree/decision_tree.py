import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.rules = []

        pass
    
    def fit(self, X, y, rules=[]):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        total_entropy = entropy(np.array(y.value_counts().tolist()))
        columns_infogain = np.zeros(len(X.columns))

        for i, column in enumerate(X):
            column_infogain = total_entropy

            for unique_value in X[column].unique():
                indices = X[X[column] == unique_value].index.tolist()
                counts = np.array(y.reindex(indices).value_counts().tolist())
                value_entropy = entropy(counts)
                prop = len(indices)/len(X.index)
                column_infogain -= prop*value_entropy

            columns_infogain[i] = column_infogain

        index_of_highest_infogain = np.argmax(column_infogain)
        root = X.iloc[:,index_of_highest_infogain]

        rules_backup = tuple(rules)

        for unique in root.unique():

            rules = list(rules_backup)

            rules.append((root.name, unique))

            indices = root[root == unique].index.tolist()
            counts = np.array(y.loc[indices].value_counts().tolist())
            unique_entropy = entropy(counts)

            if unique_entropy == 0:
                to_append = (list(tuple(rules)), y[indices].iloc[0])
                self.rules.append(to_append)
                rules.pop()
                continue
            else:
                X_new = X.loc[indices].drop(root.name,axis=1)
                y_new = y[indices]
                self.fit(X_new,y_new,rules)

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        results = []

        rules = self.get_rules()

        for i in range(len(X)):
            # print(X.iloc[i])
            data = []
            for j in range(len(X.iloc[i])):
                data.append((X.columns[j],X.iloc[i,j]))
            for k in range(len(rules)):
                if is_contained_within(data,rules[k][0]):
                    results.append(rules[k][1])

        return np.array(results)
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        return self.rules

def is_contained_within(a,b):
    a_set = set(a)

    for item in b:
        if item not in a_set:
            return False

    return True

# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



