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

        print("--------------------------------------")
        print("self.rules is currently:", self.rules)

        # print(X)
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

            print("currently processing:", root.name, ":", unique)
            print("rules_backup is", rules_backup)
            print("self.rules is", self.rules)

            print("rules before backup:", rules)

            rules = list(rules_backup)

            print("rules after backup:", rules)

            rules.append((root.name, unique))
            print("rules_backup is now", rules_backup)
            # print("RULES BEFORE", rules)
            # print("THIS VALUE IS", unique)

            indices = root[root == unique].index.tolist()
            # print(indices)
            # print(y.iloc[indices])
            # print("y:", y)
            # print("indices:", indices)
            # print("attempting to find", indices, "in", y)
            # print("result:", y.loc[indices].value_counts())
            counts = np.array(y.loc[indices].value_counts().tolist())
            # print("counts:", counts)
            unique_entropy = entropy(counts)
            # print(unique_entropy)
            if unique_entropy == 0:
                # print("y:", y)
                # print("indices:", indices)
                # rules = (rules, y[indices].iloc[0])
                # print("entropy is zero")
                # print("self.rules currently", self.rules)
                # print("adding", (rules, y[indices].iloc[0]))
                print("entropy is zero, rules are ", rules)
                print("self.rules is ", self.rules)
                print("adding", (rules, y[indices].iloc[0]), "to self.rules")
                to_append = (list(tuple(rules)), y[indices].iloc[0])
                self.rules.append(to_append)
                # self.rules.append((rules, y[indices].iloc[0]))
                print("self.rules is now", self.rules)
                rules.pop()
                print("entropy is zero")
                print("self.rules is now", self.rules)
                # print("entropy is zero, rules are now", self.rules)

                # print("ENTROPY ZERO")
                # print(unique)
                # print(rules)
                # rules.pop()
                continue
            else:
                # print("locating indices", indices)
                # print("ENTROPY NONZERO")
                # print(unique)
                # print(rules)
                X_new = X.loc[indices].drop(root.name,axis=1)
                y_new = y[indices]
                # print("hello")
                print("entropy is nonzero")
                print("self.rules is currently", self.rules)
                # print(rules)
                # self.fit(X_new,y_new,rules
                print("going deeper. rules_backup is currently", rules_backup)
                # rules_backup.pop()
                self.fit(X_new,y_new,rules)
                # print("hello")
                print("Final rules:", self.rules)
                print("///////////////////////////")

        # print("hello")
        # print(rules)


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
        # TODO: Implement 
        raise NotImplementedError()
    
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



