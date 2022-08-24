"""
main.py Program
=============================================
mendefinisikan fungsi yang digunakan dalam "classification"
"""

import os
import sys
from tempfile import gettempdir
from subprocess import call
import matplotlib.pyplot as plt # Module yang digunakan untuk plotting
import numpy as np # Modul yang menyederhanakan komputasi pada matriks
from scipy.signal import butter, lfilter, lfilter_zi # Modul pengolahan sinyal
import csv # Modul yang digunakan untuk merekam data ke dalam bentuk .csv
import time # Modul akses manipulasi waktu
import datetime as dt # Modul akses tanggal, hari, waktu saat ini
import pandas
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

NOTCH_B, NOTCH_A = butter(4, np.array([55, 65])/(256/2), btype='bandstop')


def plot_multichannel(data, params=None):
    """Membuat plot data multikanal 
    Args:
        data (numpy.ndarray):  Multichannel Data [n_samples, n_channels]
        params (dict): information about the data acquisition device

    TODO Receive labels as arguments
    """
    fig, ax = plt.subplots()

    n_samples = data.shape[0]
    n_channels = data.shape[1]

    if params is not None:
        fs = params['sampling frequency']
        names = params['names of channels']
    else:
        fs = 1
        names = [''] * n_channels

    time_vec = np.arange(n_samples) / float(fs)

    data = np.fliplr(data)
    offset = 0
    for i_channel in range(n_channels):
        data_ac = data[:, i_channel] - np.mean(data[:, i_channel])
        offset = offset + 2 * np.max(np.abs(data_ac))
        ax.plot(time_vec, data_ac + offset, label=names[i_channel])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    plt.legend()
    plt.draw()


def epoch(data, samples_epoch, samples_overlap=0):
    """ Mengekstrak epoch dari time series.
    Given a 2D array of the shape [n_samples, n_channels]
    Creates a 3D array of the shape [length_samples, n_channels, n_epochs]

    Args:
        data (numpy.ndarray or list of lists): data [n_samples, n_channels]
        samples_epoch (int): window length in samples
        samples_overlap (int): Overlap between windows in samples

    Returns:
        (numpy.ndarray): epoched data of shape
    """

    if isinstance(data, list):
        data = np.array(data)

    n_samples, n_channels = data.shape

    samples_shift = samples_epoch - samples_overlap

    n_epochs =  int(np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)

    # Markers indicate where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)

    # Divide data in epochs
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))

    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]

    return epochs


def compute_feature_vector(eegdata, fs):
    """ Ekstraksi feature dari EEG
    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata

    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    
    ### Komputasi power spectral density (PSD) ###
    winSampleLength, nbCh = eegdata.shape

    # Aplikasi Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T*w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0)/winSampleLength
    PSD = 2*np.abs(Y[0:int(NFFT/2), :])
    f = fs/2*np.linspace(0, 1, int(NFFT/2))

    # SPECTRAL FEATURES
    # Average of band powers
    #Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)
    
    # Gamma 30-44
    ind_gamma, = np.where((f >= 30) & (f < 44))
    meanGamma = np.mean(PSD[ind_gamma, :], axis=0)
    

    feature_vector = np.concatenate((meanBeta, meanGamma), axis=0)


    feature_vector = np.log10(feature_vector)

    return feature_vector


def nextpow2(i):
    """
    Mencari next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def compute_feature_matrix(epochs, fs):
    """
    Memanggil compute_feature_vector untuk setiap EEG epoch
    """
    n_epochs = epochs.shape[2]

    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_feature_vector(epochs[:, :, i_epoch], fs).T
            feature_matrix = np.zeros((n_epochs, feat.shape[0])) # Initialize feature_matrix

        feature_matrix[i_epoch, :] = compute_feature_vector(
                epochs[:, :, i_epoch], fs).T

    return feature_matrix


def train_classifier(model, feature_matrix_0, feature_matrix_1, feature_matrix_2, feature_matrix_3, feature_matrix_4):
    """
    Melatih classsifier. Pertama melakukan Z-score normalisasi, lalu fit

    Args:
        feature_matrix_0 (numpy.ndarray): array of shape (n_samples,
            n_features) with examples for Class 0
        feature_matrix_0 (numpy.ndarray): array of shape (n_samples,
            n_features) with examples for Class 1
        alg (str): Type of classifer to use. Currently only SVM is
            supported.

    Returns:
        (sklearn object): trained classifier (scikit object)
        (numpy.ndarray): normalization mean
        (numpy.ndarray): normalization standard deviation
    """
    
    # Membuat vektor Y (label kelas)
    class0 = np.full((feature_matrix_0.shape[0], 1),0)
    class1 = np.full((feature_matrix_1.shape[0], 1),1)
    class2 = np.full((feature_matrix_2.shape[0], 1),2)
    class3 = np.full((feature_matrix_3.shape[0], 1),3)
    class4 = np.full((feature_matrix_4.shape[0], 1),4)
    
    print('\nclass=\n')
    print(class0.shape)


    # Menggabungkan matriks feature dan labelnya masing-masing
    y = np.concatenate((class0, class1, class2, class3, class4), axis=0)
    features_all = np.concatenate((feature_matrix_0, feature_matrix_1, feature_matrix_2, feature_matrix_3, feature_matrix_4), axis=0)
    
    print('\ny=\n')
    print(y.shape)
    print('\nfeatures=\n')
    print(features_all.shape)

    # Normalisasi features columnwise
    mu_ft = np.mean(features_all, axis=0)
    std_ft = np.std(features_all, axis=0)

    X_train = (features_all - mu_ft) / std_ft
    
    print('\nX=\n')
    print(X_train.shape)
    
#    dataframeX = pandas.read_csv("dataX.csv", header=None)
#    datasetX = dataframeX.values
#    X_train = datasetX.astype(float)
#    
#    dataframeY = pandas.read_csv("dataY3.csv", header=None)
#    datasetY = dataframeY.values
#    y = datasetY

    np.savetxt("dataX4.csv", X_train,  delimiter=",")
    np.savetxt("dataY4.csv", y,  delimiter=",")

    # Melatih JST menggunakan default parameters
    encoder = LabelEncoder()
    encoder.fit(y.ravel())
    encoded_Y = encoder.transform(y.ravel())
#    print('\nEncoded_Y=\n')
#    print(encoded_Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_train = np_utils.to_categorical(encoded_Y)
#    print('\ndummy_y\n')
#    print(dummy_y)
    model.fit(X_train, Y_train, epochs=500, batch_size=5, shuffle=1, verbose=1)
    score = model.evaluate(X_train, Y_train, verbose=0)
    # save model and architecture to single file
    model.save("model.h5")

    #Merekam training klasifikasi
    
#    today1 = dt.datetime.today().strftime('Training''_''%H'':''%M''_''%d''-''%m''-''%Y''.csv')
#    with open(today1, "a") as file:
#        writer = csv.writer(file, delimiter = ",")
#        writer.writerow(['Classification Label','True Label',str(score * 100) + '% correctly predicted'])
#        writer.writerow([trainpred,y])
#        file.flush()
        
    # Visualisasi batas keputusan
    # plot_classifier_training(clf, X, y, features_to_plot=[0, 1])

    return mu_ft, std_ft, score[1], encoder


def test_classifier(model, feature_vector, mu_ft, std_ft, encoder):
    """ Menguji pengklasifikasi pada data baru.

    Args:
        clf (sklearn object): trained classifier
        feature_vector (numpy.ndarray): array of shape (n_samples,
            n_features)
        mu_ft (numpy.ndarray): normalization mean
        std_ft (numpy.ndarray): normalization standard deviation

    Returns:
        (numpy.ndarray): decision of the classifier on the data points
    """

    # Normalisasi feature_vector
    x = (feature_vector - mu_ft) / std_ft
    predictions = model.predict_classes(x)
    print(predictions)
    #print(encoder.inverse_transform(predictions))
    #print(predictions.shape[0])
#    threshold=0.7
#    predictions=np.where(predictions > threshold, 1, 0)
#    print(predictions)
#    print('\nTresholded=\n')
#    if predictions==np.array([0, 0, 0, 0, 0]):
#        predictions= np.array([1, 0, 0, 0, 0])
#        print(predictions)
#    print(encoder.inverse_transform(predictions))
#    print(predictions)
    y_hat = encoder.inverse_transform(predictions)

    return y_hat


def beep(waveform=(79, 45, 32, 50, 99, 113, 126, 127)):
    """ Menjalankan bunyi beep.
    Tidak membutuhkan file suara
    """
    wavefile = os.path.join(gettempdir(), "beep.wav")
    if not os.path.isfile(wavefile) or not os.access(wavefile, os.R_OK):
        with open(wavefile, "w+") as wave_file:
            for sample in range(0, 300, 1):
                for wav in range(0, 8, 1):
                    wave_file.write(chr(waveform[wav]))
    if sys.platform.startswith("linux"):
        return call("chrt -i 0 aplay '{fyle}'".format(fyle=wavefile),
                    shell=1)
    if sys.platform.startswith("darwin"):
        return call("afplay '{fyle}'".format(fyle=wavefile), shell=True)
    if sys.platform.startswith("win"):  # FIXME: This is Ugly.
        return call("start /low /min '{fyle}'".format(fyle=wavefile),
                    shell=1)


def get_feature_names(ch_names):
    """Membuat nama dari features.
    Args:
        ch_names (list): electrode names

    Returns:
        (list): feature names
    """
    bands = ['beta','gamma']

    feat_names = []
    for band in bands:
        for ch in range(len(ch_names)):
            feat_names.append(band + '-' + ch_names[ch])

    return feat_names


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """Mengabungkan "new_data" ke "data_buffer", dan mengembalikan sebuah array dengan ukuran yang sama dengan
    "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """Mendapatkan dari "buffer_array", "newest samples" (N baris dari bawah buffer)"""
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer


class DataPlotter():
    """Kelas untuk membuat dan mengupdate line plot."""

    def __init__(self, nbPoints, chNames, fs=None, title=None):
        """Inisialisasi figure."""

        self.nbPoints = nbPoints
        self.chNames = chNames
        self.nbCh = len(self.chNames)

        self.fs = 1 if fs is None else fs
        self.figTitle = '' if title is None else title

        data = np.empty((self.nbPoints, 1))*np.nan
        self.t = np.arange(data.shape[0])/float(self.fs)

        # Create offset parameters for plotting multiple signals
        self.yAxisRange = 100
        self.chRange = self.yAxisRange/float(self.nbCh)
        self.offsets = np.round((np.arange(self.nbCh)+0.5)*(self.chRange))

        # Create the figure and axis
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_yticks(self.offsets)
        self.ax.set_yticklabels(self.chNames)

        # Initialize the figure
        self.ax.set_title(self.figTitle)

        self.chLinesDict = {}
        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName], = self.ax.plot(
                    self.t, data+self.offsets[i], label=chName)

        self.ax.set_xlabel('Time')
        self.ax.set_ylim([0, self.yAxisRange])
        self.ax.set_xlim([np.min(self.t), np.max(self.t)])

        plt.show()

    def update_plot(self, data):
        """ Update plot """

        data = data - np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        std_data[np.where(std_data == 0)] = 1
        data = data/std_data*self.chRange/5.0

        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(data[:, i] + self.offsets[i])

        self.fig.canvas.draw()

    def clear(self):
        """ Mengclear figure """

        blankData = np.empty((self.nbPoints, 1))*np.nan

        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(blankData)

        self.fig.canvas.draw()

    def close(self):
        """ Menutup figure """

        plt.close(self.fig)


def plot_classifier_training(clf, X, y, features_to_plot=[0, 1]):
    """Visualisasi batas pengklasifikasi.

    Args:
        clf (sklearn object): trained classifier
        X (numpy.ndarray): data to visualize the decision boundary for
        y (numpy.ndarray): labels for X

    Keyword Args:
        features_to_plot (list): indices of the two features to use for
            plotting
    """

    plot_colors = "bry"
    plot_step = 0.02
    n_classes = len(np.unique(y))

    x_min = np.min(X[:, 1])-1
    x_max = np.max(X[:, 1])+1
    y_min = np.min(X[:, 0])-1
    y_max = np.max(X[:, 0])+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)

    plt.axis('tight')
    