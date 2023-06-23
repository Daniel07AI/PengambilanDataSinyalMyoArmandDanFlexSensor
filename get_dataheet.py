import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import myo
from threading import Lock, Thread
from scipy import signal

# --------------------------------------------------------------------------------------
# PROGRAM PENGAMBIAN DATA KEKUATAN
# --------------------------------------------------------------------------------------
ser = serial.Serial('com6',baudrate=2000000)
data = []
start_time = time.time()
data_final=[]
seconds=48
collectt=True

def flex():
    global collectt
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time-start_time
        
        arduino_data = ser.readline().decode('ascii', errors = 'ignore')
        arduino_data = arduino_data.strip().split(" ")
        # data.append(arduino_data)
        if elapsed_time >= 1:
            break
    time.sleep(2)
    start_time = time.time()
    print('=========Mulai==========\n')
    # time=deque(maxlen=1)
    while collectt:
        current_time = time.time()
        elapsed_time = current_time-start_time
        
        arduino_data = ser.readline().decode('ascii', errors = 'ignore')
        arduino_data = arduino_data.strip().split(" ")
        data.append(arduino_data)
        t=int(elapsed_time)
        print(t, end='\r')
        if t ==0 or t ==8 or t ==16 or t ==24 or t ==32 or t ==40 :
            print('     |=====> GENGGAM <=====  ',end='\r')
        elif t==3 or t ==11 or t ==19 or t ==27 or t ==35 or t ==43:
            print('     |=====> LEPASKAN <===== ',end='\r')
        elif t==5 or t ==13 or t ==21 or t ==29 or t ==37 or t ==45:
            print('     |=====> ISTIRAHAT <=====',end='\r')
        
        if elapsed_time >= seconds:
            collectt=False
            break
    data2=pd.DataFrame(data)
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    data2.to_csv('Kekuatan.csv', index=False) #<------------NAMA FILE FLEX----------
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    print(elapsed_time)
    print("========While Loop Berhenti=========")
# -----------------------------------------------------------------------------------
# PROGRAM PENGAMBILAN DATA EMG (MYO ARMBAND)
# -----------------------------------------------------------------------------------
class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # myo.DeviceListener

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))

class Plot(object):
    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        self.start_time=time.time()
        self.curent_time=time.time()
        self.elapsetime=self.curent_time-self.start_time
        self.collect=True
        self.seconds=50
        self.fig = plt.figure()
        self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        [(ax.set_ylim([-100, 100])) for ax in self.axes]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        plt.ion()

    def update_plot(self):
        emg_data = self.listener.get_emg_data()
        emg_data = np.array([x[1] for x in emg_data]).T
        save_emg = pd.DataFrame(emg_data)
        save_emgTranspose = save_emg.T
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        save_emgTranspose.to_csv('EMG.csv',index=False) #<--------NAMA FILE EMG----
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        for g, data in zip(self.graphs, emg_data):
            if len(data) < self.n:
                # Fill the left side with zeroes.
                data = np.concatenate([np.zeros(self.n - len(data)), data])
            g.set_ydata(data)

        plt.draw()
    def main(self):
        global collectt
        while collectt:
            self.update_plot()
            plt.pause(1.0 / 30)
def main():
    myo.init(bin_path='C:\\Users\\asus A442U\\Documents\\myo-sdk-win-0.9.0\\myo-sdk-win-0.9.0\\bin')
    hub = myo.Hub()
    listener = EmgCollector(10000)
    with hub.run_in_background(listener.on_event):
        Plot(listener).main()
        
# -------------------------------------------------------------------------------------
# RUN PROGRAM PENGAMBILAN DATA EMG DAN KEKUTAN UNTUK REGRESI
# -------------------------------------------------------------------------------------

def get_data():
    p1 =Thread(target=flex)
    p2 =Thread(target=main)

    p1.start()
    p2.start() 
    p1.join()
    p2.join() 
def resample():
    data_Grip=pd.read_csv('Kekuatan.csv')
    data_Myo=pd.read_csv('EMG.csv')
    index=[]
    a=0

    Grip=np.array(data_Grip)
    Myo=np.array(data_Myo)

    # data_skala = MinMaxScaler(feature_range=(0, 10), copy=True)
    # data_skala1 = data_skala.fit_transform(Grip)

    # n_Grip=len(data_Grip)
    # n_myo=len(data_Myo)

    # print(n_myo)
    # ---------------------------------------------------------------
    downsample1=signal.resample(Grip,9600) 
    downsample2=signal.resample(Myo,9600)#resample data flex
    n_resm1=len(downsample1)
    n_resm2=len(downsample2)

    print(n_resm1)
    print(n_resm2)

    # ----------------------------------------------------------------

    dataMyo=deque(maxlen=n_resm2-a)

    for i in range(n_resm2):
        dataMyo.append(downsample2[i])
    np.array(dataMyo)

    for i in range(n_resm1-a,n_resm1):
        index.append(i)
    dataGrip=np.delete(downsample1,index, axis=0)

    Grip_=pd.DataFrame(dataGrip)
    Myo_=pd.DataFrame(dataMyo)

    datasheet=pd.concat([Myo_,Grip_], axis=1)
    datasheet.to_csv('responden-2.csv',index=False)

if __name__=='__main__':
    get_data()
    resample()
        



        
