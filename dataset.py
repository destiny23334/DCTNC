from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def cal_noisy_ratio(integrate_labels: list, true_labels: np.ndarray):
    errors = 0
    for i, idx_ in enumerate(integrate_labels):
        if integrate_labels[i] != true_labels[i]:
            errors += 1
    return errors / len(integrate_labels)


def get_noisy_label(length: int, answers: np.ndarray):
    multi_noisy_labels = []
    integrate_labels = []
    gini = []
    for i in range(length):
        noisy_label = answers[i, answers[i] >= 0]
        multi_noisy_labels.append(noisy_label)
        value, count = np.unique(noisy_label, return_counts=True)
        integrate_labels.append(value[np.argmax(count)])
        gini.append(cal_gini(count))
    return multi_noisy_labels, integrate_labels, gini


def cal_gini(x: np.ndarray) -> int:
    return 1 - ((x / x.sum()) ** 2).sum()


class MyDataset(Dataset):
    def __init__(self):
        self.integrate_labels = None

    def __getitem__(self, item):
        if self.isTruth:
            return self.data[item, :], self.true_labels[item]
        else:
            return self.data[item, :], self.integrate_labels[item]

    def __len__(self):
        return self.data.shape[0]

    def set_integrate_label(self, integrate_label):
        self.integrate_labels = np.array(integrate_label, dtype=np.int32)

    def _init(self, name, dir_path='./data/', idx=None, isTruth=False):
        self.data = pd.read_csv(dir_path + name + '/' + name + '.csv').values
        self.true_labels = self.data[:, -1]
        self.data = self.data[:, :-1]

        self.answers = pd.read_csv(dir_path + name + '/' + name + '.answers.txt', sep=' ', header=None).values
        self.num_feature = self.data.shape[1]
        self.isTruth = isTruth

        if idx is not None:
            self.data = self.data[idx, :]
            self.true_labels = self.true_labels[idx]
            self.answers = self.answers[idx, :]

        self.multi_noisy_labels, self.integrate_labels, self.gini = \
            get_noisy_label(self.data.shape[0], self.answers)

        self.num_classes = int(max(self.true_labels)) + 1


class LabelMe(MyDataset):
    def __init__(self, dir_path='./data/', idx=None, isTruth=False):
        super().__init__()
        super()._init(name='labelme', dir_path=dir_path, idx=idx, isTruth=isTruth)


class BalanceScale(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(BalanceScale, self).__init__()
        super()._init(name='balance-scale', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Biodeg(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Biodeg, self).__init__()
        super()._init(name='biodeg', dir_path=dir_path, idx=idx, isTruth=isTruth)


class BreastW(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super().__init__()
        super()._init(name='breast-w', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Diabetes(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Diabetes, self).__init__()
        super()._init(name='diabetes', dir_path=dir_path, idx=idx, isTruth=isTruth)


class HeartStatlog(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(HeartStatlog, self).__init__()
        super()._init(name='heart-statlog', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Ionosphere(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Ionosphere, self).__init__()
        super()._init(name='ionosphere', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Iris(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Iris, self).__init__()
        super()._init(name='iris', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Letter(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Letter, self).__init__()
        super()._init(name='letter', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Segment(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Segment, self).__init__()
        super()._init(name='segment', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Sonar(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Sonar, self).__init__()
        super()._init(name='sonar', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Spambase(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Spambase, self).__init__()
        super()._init(name='spambase', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Vehicle(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Vehicle, self).__init__()
        super()._init(name='vehicle', dir_path=dir_path, idx=idx, isTruth=isTruth)


class Waveform(MyDataset):
    def __init__(self, dir_path='./data/syn/', idx=None, isTruth=False):
        super(Waveform, self).__init__()
        super()._init(name='waveform', dir_path=dir_path, idx=idx, isTruth=isTruth)

