import sys, os
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataset(if_data_shake):
    
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, 'mnist/' + filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists('mnist/' + filename):
            download(filename)
        with gzip.open('mnist/' + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 784)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists('mnist/' + filename):
            download(filename)
        with gzip.open('mnist/' + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    if if_data_shake==0:
        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

    else:
        X_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        X_train.flags.writeable = True
        y_train.flags.writeable = True

        for i in range(50000):
            y_train[i] = i%2
            if i%2 ==0:
                dd = -10
            else:
                dd = 10
            mm = np.ones((28,28))
            x_train[i] = mm*dd

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_cifar(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    
    X_train, y_train = d['data'], np.array(d['labels'], dtype=np.int32)

    X_train = X_train.reshape(-1, 3, 32, 32)

    X_train, X_val = X_train[:-1000], X_train[-1000:]
    y_train, y_val = y_train[:-1000], y_train[-1000:]
    
    X_train, X_test = X_train[:-1000], X_train[-1000:]
    y_train, y_test = y_train[:-1000], y_train[-1000:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_20news():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(newsgroups_train.data)
    X_test = vectorizer.transform(newsgroups_test.data)

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    
    X_train, X_val = X_train[:-1000], X_train[-1000:]
    y_train, y_val = y_train[:-1000], y_train[-1000:]

    return X_train, y_train, X_val, y_val, X_test, y_test
