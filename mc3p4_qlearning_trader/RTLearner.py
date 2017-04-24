"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import math

class DecisionNode():
    def __init__(self, left, right, decision_function, val=None):
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.val = val

    def decide(self, feature):
        """Return on a label if node is leaf,or pass the decision down to the node's left/right child (depending on decisionfunction)."""
        if self.val is not None:
            return self.val
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.total_leaves = 0
        self.root = None

    def get_total_leaves(self):
        return self.total_leaves

    def build_tree(self, dataX, dataY):
        if len(dataY) <= self.leaf_size:
            self.total_leaves = self.total_leaves + 1
            return DecisionNode(None, None, None, val=np.average(dataY))

        # SPLIT DATA
        selected_feature = np.random.randint(dataX.shape[1])
        selected_rows = np.random.choice(np.arange(dataX.shape[0]), size=2, replace=False)
        selected_data1 = dataX[selected_rows[0], selected_feature]
        selected_data2 = dataX[selected_rows[1], selected_feature]
        split_val = (selected_data1 + selected_data2) / 2.
        dataX_left = np.asarray([j for i, j in zip(dataX[:, selected_feature], dataX) if i <= split_val])
        dataY_left = np.asarray([j for i, j in zip(dataX[:, selected_feature], dataY) if i <= split_val])
        dataX_right = np.asarray([j for i, j in zip(dataX[:, selected_feature], dataX) if i > split_val])
        dataY_right = np.asarray([j for i, j in zip(dataX[:, selected_feature], dataY) if i > split_val])

        # EXCEPTION HANDLING: a tie on split_val
        if dataX_left.shape[0] == 0 or dataX_right.shape[0] == 0:
            dataX_left = dataX[0:1, :]
            dataY_left = dataY[0:1]
            dataX_right = dataX[1:, :]
            dataY_right = dataY[1:]

        return DecisionNode(self.build_tree(dataX_left, dataY_left), self.build_tree(dataX_right, dataY_right),lambda x: x[selected_feature] <= split_val, None)

    def addEvidence(self, dataX, dataY):
        self.root = self.build_tree(dataX, dataY)

    def query(self, points):
        return [self.root.decide(example) for example in points]

    def author(self):
        return 'llee81'


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
