"""
main.py program
=============================================
Description:
Program ini merupakan program klasifikasi dan penerjemahan gelombang otak. Program mengklasifikasikan 4 gerakan motorik menjadi 4 kelas, dan melakukan prediksi data secara realtime. Gelombang otak yang digunakan yaitu gelombang beta (12-30Hz) dan gamma (30-44Hz).Program menerima dan menstream EEG Data yang dideteksi Muse EEG 2016 (menggunakan 4 channels : TP9, AF7, AF8, TP10).
"""

import argparse
import numpy as np  # Modul yang menyederhanakan komputasi pada matriks
import matplotlib.pyplot as plt  # Module yang digunakan untuk plotting
from pylsl import StreamInlet, resolve_byprop  # Module untuk menerima EEG data
import serial #Module untuk komunikasi serial dengan Arduino
import utils  # Modul yang mendefinisikan berbagai fungsi yang dibutuhkan
import csv # Modul yang digunakan untuk merekam data ke dalam bentuk .csv
import time # Modul akses manipulasi waktu
import datetime as dt # Modul akses tanggal, hari, waktu saat ini
import pandas
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":

    """ 0. PARSE ARGUMENTS """
    parser = argparse.ArgumentParser(description='Classification Program')
    parser.add_argument('channels', metavar='N', type=int, nargs='*',
        default=[0, 1, 2, 3],
        help='channel number to use. If not specified, all the channels are used')

    args = parser.parse_args()

    """ 1. Menghubungkan Dengan EEG"""

    # Mencari LSL stream yang aktif
    print('Mencari EEG stream')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Tidak dapat menemukan EEG stream.')

    # Mengatur aktif EEG stream ke inlet dan mengaplikasikan koreksi waktu
    print("Mulai memperoleh data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Mendapatkan info, deskripsi, frekuensi sampling, dan jumlah kanal stream
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()

    # Mendapatkan nama semua kanal
    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, n_channels):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    """ 2. Mengatur Parameter Eksperimental """

    # Panjang dari buffer EEG data (dalam detik)
    # Buffer ini akan menahan n detik data terakhir yang digunakan untuk kalkulasi 
    buffer_length = 15

    # Panjang epoch yang digunakan untuk komputasi FFT (dalam detik)
    epoch_length = 1

    # Jumlah overlap antara 2 epoch berurutan (dalam detik )
    overlap_length = 0.2

    # Jumlah untuk menggeser permulaaan dari setiap epoch berurutan
    shift_length = epoch_length - overlap_length

    # Index kanal yang digunakan (electroda)
    # 0 = Telinga kiri, 1 = Dahi depan kiri, 2 = Dahi depan kanan, 3 = Telinga kanan
    index_channel = args.channels
    
    # Nama kanal untuk tujuan plotting 
    ch_names = [ch_names[i] for i in index_channel]
    n_channels = len(index_channel)

    # Mendapatkan mana fitur 
    # ex. ['delta - CH1', 'pwr-theta - CH1', 'pwr-alpha - CH1',...]
    feature_names = utils.get_feature_names(ch_names)

    # Lamanya waktu pengumpulan data training untuk 1 kelas (dalam detik)
    training_length = 1
    istirahat_length = 1

    """ 3. Merekam Data Training"""
    # Bersiap
    print('\nBersiap dalam 10 detik!\n')
    utils.beep()

    time.sleep(istirahat_length)
    
    # Merekam data kelas 0
    print('\nGerakkan Bibir ke Kanan!\n')
    
    utils.beep()
    utils.beep()
    eeg_data0, timestamps0 = inlet.pull_chunk(
            timeout=training_length+1, max_samples=fs * training_length)
    eeg_data0 = np.array(eeg_data0)[:, index_channel]
    
    # Istirahat
    print('\nIstirahat 10 detik!\n')
    utils.beep()

    time.sleep(istirahat_length)
    
    # Merekam data kelas 1
    print('\nKedipkan Mata Terus Menerus!\n')
    utils.beep()
    utils.beep()

    eeg_data1, timestamps1 = inlet.pull_chunk(
            timeout=training_length+1, max_samples=fs * training_length)
    eeg_data1 = np.array(eeg_data1)[:, index_channel]

    # Istirahat
    print('\nIstirahat 10 detik!\n')
    utils.beep()

    time.sleep(istirahat_length)

    # Merekam data kelas 2
    print('\nMiringkan Kepala ke Kiri!\n')
    utils.beep()
    utils.beep()

    eeg_data2, timestamps1 = inlet.pull_chunk(
            timeout=training_length+1, max_samples=fs * training_length)
    eeg_data2 = np.array(eeg_data2)[:, index_channel]
    
    # Istirahat
    print('\nIstirahat 10 detik!\n')
    utils.beep()

    time.sleep(istirahat_length)
    
    # Merekam data kelas 3
    print('\nKuatkan Rahang dengan Menggigit!\n')
    utils.beep()
    utils.beep()
    
    eeg_data3, timestamps1 = inlet.pull_chunk(
            timeout=training_length+1, max_samples=fs * training_length)
    eeg_data3 = np.array(eeg_data3)[:, index_channel]
    
    # Istirahat
    print('\nIstirahat 10 detik!\n')
    utils.beep()

    time.sleep(istirahat_length)
    
    # Merekam data kelas 4
    print('\nGerakkan Kepala ke Atas!\n')
    utils.beep()
    utils.beep()
    
    eeg_data4, timestamps1 = inlet.pull_chunk(
            timeout=training_length+1, max_samples=fs * training_length)
    eeg_data4 = np.array(eeg_data4)[:, index_channel]
    
    utils.beep()
     
    # Membagi data menjadi epoch
    eeg_epochs0 = utils.epoch(eeg_data0, epoch_length * fs,
                             overlap_length * fs)
    eeg_epochs1 = utils.epoch(eeg_data1, epoch_length * fs,
                             overlap_length * fs)
    eeg_epochs2 = utils.epoch(eeg_data2, epoch_length * fs,
                             overlap_length * fs)
    eeg_epochs3 = utils.epoch(eeg_data3, epoch_length * fs,
                             overlap_length * fs)
    eeg_epochs4 = utils.epoch(eeg_data4, epoch_length * fs,
                             overlap_length * fs)


    """ 4. Komputasi Features dan Melatih Pengklasifikasi """

    feat_matrix0 = utils.compute_feature_matrix(eeg_epochs0, fs)
    feat_matrix1 = utils.compute_feature_matrix(eeg_epochs1, fs)
    feat_matrix2 = utils.compute_feature_matrix(eeg_epochs2, fs)
    feat_matrix3 = utils.compute_feature_matrix(eeg_epochs3, fs)
    feat_matrix4 = utils.compute_feature_matrix(eeg_epochs4, fs)

    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=8, init='normal', activation='relu'))
    model.add(Dense(15, init='normal', activation='relu'))
    model.add(Dense(5, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    [mu_ft, std_ft, score, encoder] = utils.train_classifier(model,
            feat_matrix0, feat_matrix1, feat_matrix2, feat_matrix3, feat_matrix4)

    print(str(score * 100) + '% terprediksi secara benar')

    utils.beep()
    utils.beep()
    utils.beep()

    """ 5. Menggunakan Pengklasifikasi Secara Realtime """

    # Inisialisasi buffer untuk menyimpan raw EEG dan keputusan 
    eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
    filter_state = None  # for use with the notch filter
    decision_buffer = np.zeros((30, 1))

    plotter_decision = utils.DataPlotter(30, ['Decision'])

    # Struktur try/except memungkinkan untuk quit loop dengan membatalkan script dengan <Ctrl-C>
    print('Ctrl-C in the console to break the while loop.')

    try:
        while True:

            """ 3.1 Mendapatkan Data """
            # Mendapatkan EEG data dari LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(shift_length * fs))

            # Hanya menyimpan kanal yang diinginkan
            ch_data = np.array(eeg_data)[:, index_channel]

            # Update EEG buffer
            eeg_buffer, filter_state = utils.update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

            """ 3.2 Komputasi Features dan klasifikasi """
            # Mendapatkan sampel terbaru dari buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                            epoch_length * fs)

            # Komputasi feature
            feat_vector = utils.compute_feature_vector(data_epoch, fs)
            y_hat = utils.test_classifier(model, feat_vector.reshape(1, -1), mu_ft, std_ft, encoder)
            hasil=y_hat.tostring()
            today2 = dt.datetime.today().strftime('Classification''_''%H'':''%M''_''%d''-''%m''-''%Y''.csv')
            print(y_hat)
            
            #Merekam pengklasifikasian dalam bentuk .csv
            with open(today2, "a") as file:
                writer = csv.writer(file, delimiter = ",")
                writer.writerow([y_hat])
                file.flush()
            
            
            #decision_buffer, _ = utils.update_buffer(decision_buffer, np.reshape(y_hat, (-1, 1)))
#            ser=serial.Serial('/dev/ttyUSB0',9600)
#            ser.write(hasil)    

    except KeyboardInterrupt:
        print('Closed!')