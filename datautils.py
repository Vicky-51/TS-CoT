
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch


def load_HAR():
    data_path = "./data/HAR/"
    train_ = torch.load(data_path + "train.pt")
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_EEG():
    data_path = "./data/sleepEDF/"
    train_ = torch.load(data_path + "train.pt")
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y) # [N, L, C]
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_Epi():
    data_path = "./data/epilepsy/"
    train_ = torch.load(data_path + "train.pt")
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_Waveform():
    data_path = "./Waveform/"
    train_ = torch.load(data_path + "train.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    test = test_['samples']
    # test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train
    train_y = train_labels
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_HAR_fft():
    data_path = "./data/HAR/"
    train_ = torch.load(data_path + "train.pt")
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples'] # [5881, 9, 128]
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_EEG_fft():
    data_path = "./data/sleepEDF/"
    train_ = torch.load(data_path + "train.pt")
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_Epi_fft():
    data_path = "./data/epilepsy/"
    train_ = torch.load(data_path + "train.pt")
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)


    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_Waveform_fft():
    data_path = "./data/Waveform/"
    train_ = torch.load(data_path + "train.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train
    train_y = train_labels
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X).abs()
    test_X_fft = torch.fft.fft(test_X).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)


    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_HAR_two_view():
    train_X, train_y, test_X, test_y = load_HAR()
    train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    return [train_X, train_X_fft], train_y, [test_X, test_X_fft], test_y

def load_EEG_two_view():
    train_X, train_y, test_X, test_y = load_EEG()
    train_X_fft, _, test_X_fft, _ = load_EEG_fft()
    return [train_X, train_X_fft], train_y, [test_X, test_X_fft], test_y

def load_Epi_two_view():
    train_X, train_y, test_X, test_y = load_Epi() # train_X: 9200, 178, 1, train_y: 9200,
    train_X_fft, _, test_X_fft, _ = load_Epi_fft()
    return [train_X, train_X_fft], train_y, [test_X, test_X_fft], test_y

def load_Waveform_two_view():
    train_X, train_y, test_X, test_y = load_Waveform() # train_X: 9200, 178, 1, train_y: 9200,
    train_X_fft, _, test_X_fft, _ = load_Waveform_fft()
    return [train_X, train_X_fft], train_y, [test_X, test_X_fft], test_y



