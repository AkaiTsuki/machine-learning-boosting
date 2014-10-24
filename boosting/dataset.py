__author__ = 'jiachiliu'

import numpy as np
from boosting.preprocessing import Encoder, Imputer


class CsvFileReader:
    """
    CsvFileReader will read data from csv file
    """

    def __init__(self, path):
        self.path = path

    def read(self, delimiter, converter):
        f = open(self.path)
        lines = f.readlines()
        return self.parse_lines(lines, delimiter, converter)

    @staticmethod
    def parse_lines(lines, delimiter, converter):
        data = []
        for line in lines:
            if line.strip():
                row = [s.strip() for s in line.strip().split(delimiter) if s.strip()]
                data.append(row)

        return np.array(data, converter)


def load_spambase():
    reader = CsvFileReader('data/spambase.data')
    data = reader.read(',', float)
    total_col = data.shape[1]
    return data[:, :total_col - 1], data[:, total_col - 1]


def load_boston_house():
    reader = CsvFileReader('data/housing_train.txt')
    train_data = reader.read(' ', float)
    train = train_data[:, :train_data.shape[1] - 1]
    train_target = train_data[:, train_data.shape[1] - 1]

    test_data = CsvFileReader('data/housing_test.txt').read(' ', float)
    test = test_data[:, :test_data.shape[1] - 1]
    test_target = test_data[:, test_data.shape[1] - 1]

    return train, train_target, test, test_target


def load_perceptron():
    reader = CsvFileReader('data/perceptronData.txt')
    data = reader.read('\t', float)
    total_col = data.shape[1]
    return data[:, :total_col - 1], data[:, total_col - 1]


def load_2gaussian():
    reader = CsvFileReader('data/2gaussian.txt')
    data = reader.read(' ', float)
    return data


def load_3gaussian():
    reader = CsvFileReader('data/3gaussian.txt')
    data = reader.read(' ', float)
    return data


def load_vote():
    reader = CsvFileReader('data/vote/vote.data')
    vote = reader.read("\t", 'a25')

    encoder = Encoder()
    encoder.fit(vote, range(16))

    label_code = {}
    label_code['r'] = 0.0
    label_code['d'] = 1.0

    encoder.encode_label(vote, 16, label_code)
    vote = np.array(vote.tolist(), float)

    imputer = Imputer()
    imputer.fill(vote, range(16))

    total_col = vote.shape[1]
    return vote[:, :total_col - 1], vote[:, total_col - 1]


def load_crx():
    reader = CsvFileReader('data/crx/crx.data')
    data = reader.read("\t", 'a25')

    encoder = Encoder()
    encoder.fit(data, range(15))

    label_code = {}
    label_code['-'] = 0.0
    label_code['+'] = 1.0

    encoder.encode_label(data, 15, label_code)
    data = np.array(data.tolist(), float)

    imputer = Imputer()
    imputer.fill(data, range(15))

    total_col = data.shape[1]
    return data[:, :total_col - 1], data[:, total_col - 1]


class Configuration:
    """
    This class contains all information of a data set
    """

    def __init__(self, path):
        self.path = path
        self.total_instances = 0
        self.discrete_feature = 0
        self.numeric_feature = 0
        self.features = []
        self.feature_encode = {}
        self.feature_decode = {}

    def load_config(self, sep="\t"):
        """
        load the config file to get data set description
        :param sep: separator to parse each line in the file
        :return: configuration of the data set
        """
        f = open(self.path)
        lines = f.readlines()
        self.load_summary(lines[0].strip(), sep)
        total_features = self.get_total_features()
        for i in range(total_features):
            self.load_feature(lines[i + 1].strip(), sep)

        self.load_feature(lines[total_features + 1].strip(), sep)

    def load_feature(self, line, sep):
        """
        load feature description from a line based on number of words in the line
        :param line: a string
        :param sep: separator
        """
        words = line.split(sep)
        if len(words) == 1:
            self.load_numeric_feature(words)
        else:
            self.load_discrete_feature(words)

    def load_numeric_feature(self, words):
        self.features.append([float(words[0])])

    def load_discrete_feature(self, words):
        value_count = int(words[0])
        values = []
        for i in range(value_count):
            values.append(words[i + 1].strip())
        self.features.append(values)

    def load_summary(self, line, sep):
        """
        Load the summary of data set
        :param line: a string
        :param sep: a separator
        """
        words = line.split(sep)
        self.total_instances = int(words[0])
        self.discrete_feature = int(words[1])
        self.numeric_feature = int(words[2])

    def encode_discrete_feature(self):
        for i in range(len(self.features)):
            feature = self.features[i]
            if len(feature) == 1:
                continue
            encode_list = {}
            decode_list = {}
            for f in range(len(feature)):
                encode_list[feature[f]] = f
                decode_list[f] = feature[f]
            self.feature_encode[i] = encode_list
            self.feature_decode[i] = decode_list

    def get_total_features(self):
        return self.discrete_feature + self.numeric_feature

    def encode(self, data):
        """
        Encode the data
        :param data: a data set
        """
        m, n = data.shape
        for i in range(m):
            for j in range(n):
                val = data[i][j]
                if j in self.feature_encode and val in self.feature_encode[j]:
                    data[i][j] = self.feature_encode[j][val]

    def fill_missing_value(self, data):
        m, n = data.shape
        for f in range(n):
            val = self.find_majority_values(data, f)
            for t in range(m):
                if data[t][f] == '?':
                    data[t][f] = val

    def find_majority_values(self, data, f):
        values = self.features[f]
        majority = None
        majority_count = 0

        for val in values:
            current = len(data[data[:, f] == val])
            if current > majority_count:
                majority_count = current
                majority = val

        return majority






