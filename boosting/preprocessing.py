__author__ = 'jiachiliu'

import numpy as np


class Encoder:
    """
    A Encoder will fit the given data set with discrete features.
    For all values in given features, it will replace the string value to
    a int value so that the data set can fit into the np.array(float)
    """

    def __init__(self):
        self.encode_dict = {}

    def fit(self, data, features):
        """
        Replace the string value in given features to int value
        :param data: a data set
        :param features: a list of feature that needs to replaced
        """
        m, n = data.shape
        for f in features:
            for t in range(m):
                val = data[t][f]
                if self.get_code(val, f) is None:
                    self.create_code(val, f)
                data[t][f] = self.get_code(val, f)

    def encode_label(self, data, label, label_code):
        m, n = data.shape
        for t in range(m):
            data[t][label] = label_code[data[t][label]]

    def get_code(self, val, f):
        """
        Get the code of given value under feature f
        :param val: a value
        :param f: feature index
        :return: the code in string
        """
        if val == '?':
            return 'NaN'
        if f not in self.encode_dict:
            return None
        if val not in self.encode_dict[f]:
            return None
        return self.encode_dict[f][val]

    def create_code(self, val, f):
        """
        Create a new code based on feature and value
        :param val: a value
        :param f: feature index
        """
        if f not in self.encode_dict:
            self.encode_dict[f] = {}
        self.encode_dict[f][val] = len(self.encode_dict[f])


class Imputer:
    """
    A imputer will fill missing value with suitable values.
    For discrete value, it will fill with majority frequency one.
    For numeric value, it will fill with mean
    """

    def __init__(self):
        self.fit_values = {}

    def fill(self, data, features):
        """
        Fill the missing value with majority one
        :param data: data set
        :param features: discrete feature lists
        :return: self
        """
        m, n = data.shape
        for f in range(n):
            if f in features:
                self.fill_discrete_value(data, f)
            else:
                self.fill_continue_value(data, f)
        return self

    def fill_discrete_value(self, data, f):
        column = data[:, f]
        self.compute_discrete_value(column, f)

        for t in range(len(column)):
            if np.isnan(data[t][f]):
                data[t][f] = self.fit_values[f]

    def fill_continue_value(self, data, f):
        column = data[:, f]
        self.compute_numeric_value(column, f)

        for t in range(len(column)):
            if np.isnan(data[t][f]):
                data[t][f] = self.fit_values[f]

    def compute_discrete_value(self, column, f):
        counter = {}
        for v in column:
            if np.isnan(v):
                continue
            else:
                if v in counter:
                    counter[v] += 1
                else:
                    counter[v] = 1
        max_count = 0
        best_value = None
        for v, c in counter.items():
            if c > max_count:
                best_value = v
                max_count = c
        self.fit_values[f] = best_value

    def compute_numeric_value(self, column, f):
        mean = 0.0
        count = 0
        for v in column:
            if np.isnan(v):
                continue
            else:
                mean += v
                count += 1

        self.fit_values[f] = 1.0 * mean / count