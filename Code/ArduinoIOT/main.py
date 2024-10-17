import random
import threading
import warnings
from tkinter import messagebox

import mne.io
import redis
import time
import keyring
import numpy as np
import os
import serial
import serial.tools.list_ports
from scipy.signal import resample
import tkinter as tk

Com_name = 'COM4'

project_path = '../'
EEG_path = os.path.join(project_path, 'Dataset/CHB-MIT/chb01/chb01_01.edf')

max_queue_length = 50
queue_name = 'EEG'

start_point = 1
current_index = 0
EEG_epoch = np.array([])
ser = serial.Serial()
r = redis.Redis()
list_data = []
root_windows = tk.Tk()
serial_success_load = False
stop_event = threading.Event()

import tkinter as tk


def set_UI():
    root_windows.title("NeuroSafe:Realtime Monitoring System")
    root_windows.geometry("500x350")

    label = tk.Label(root_windows, text="Realtime Monitoring System", font=("Helvetica", 16))
    label.pack(pady=20)

    copyright_label = tk.Label(root_windows, text="©Ziyi & XMUM", font=("Helvetica", 10))
    copyright_label.pack(anchor='e', padx=40)

    radio_var = tk.IntVar()

    main_frame = tk.Frame(root_windows)
    main_frame.pack(padx=20, pady=20)

    fake_data_label = tk.Label(main_frame, text="Fake Data", font=("Helvetica", 14))
    fake_data_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

    fake_frame = tk.Frame(main_frame)
    fake_frame.grid(row=1, column=0, padx=10, pady=10)

    fake_radio_btn1 = tk.Radiobutton(fake_frame, text="Linear Wave", variable=radio_var, value=1,
                                     font=("Helvetica", 12))
    fake_radio_btn2 = tk.Radiobutton(fake_frame, text="Sin Wave", variable=radio_var, value=2, font=("Helvetica", 12))
    fake_radio_btn3 = tk.Radiobutton(fake_frame, text="EEG Wave (CHB-MIT)", variable=radio_var, value=3,
                                     font=("Helvetica", 12))

    fake_radio_btn1.pack(anchor=tk.W)
    fake_radio_btn2.pack(anchor=tk.W)
    fake_radio_btn3.pack(anchor=tk.W)

    real_data_label = tk.Label(main_frame, text="Real Data", font=("Helvetica", 14))
    real_data_label.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

    real_frame = tk.Frame(main_frame)
    real_frame.grid(row=1, column=1, padx=10, pady=10)

    real_radio_btn = tk.Radiobutton(real_frame, text="Real IOT Input", variable=radio_var, value=4,
                                    font=("Helvetica", 12))

    real_radio_btn.pack(anchor=tk.W)

    def on_button_click():
        def run_operation(_operation):
            try:
                if selected_value in [1, 2, 3]:
                    logic_for_fake_data(operation=_operation, stop_event=stop_event)
                elif selected_value == 4:
                    logic_for_realtime_data(stop_event=stop_event)
            except:

                messagebox.showinfo(title="ERROR", message="Lost Connect to Ziyi's Server")
                button['text'] = "Start"

        global thread
        if button['text'] == "Start":
            selected_value = radio_var.get()
            if selected_value in [1, 2, 3]:
                if selected_value == 1:
                    operation = generate_linear_curve
                elif selected_value == 2:
                    operation = generate_sin_curve
                else:
                    operation = generate_EEG_curve
            else:
                operation = None
            button['text'] = "Stop"
            thread = threading.Thread(target=run_operation, args=[operation])
            thread.start()
        else:

            stop_event.set()
            thread.join()
            root_windows.destroy()

    button = tk.Button(main_frame, text="Start", command=on_button_click, font=("Helvetica", 15), width=15)
    button.grid(row=2, column=0, columnspan=2, pady=10)


def set_redis():
    global r
    r = redis.Redis(host='101.43.195.210', port=6379, db=1, password=keyring.get_password('Coconut', 'Coconut'))
    r.delete(queue_name)


def push_to_queue(element):
    global r
    if r.llen(queue_name) >= max_queue_length:
        return False
    r.lpush(queue_name, element)
    print("push", element)
    return True


def generate_sin_curve(n_points=10, y_range=(0, 10), cycle=1.5 * np.pi):
    global start_point
    operation = random.choice(['+', '-'])

    if operation == '+':
        stop_point = start_point + cycle
    else:
        stop_point = start_point - cycle

    x = np.linspace(start_point, stop_point, n_points)
    y = (y_range[1] - y_range[0]) * np.sin(x) / 2 + (y_range[1] + y_range[0]) / 2
    str_y = ' '.join(map(str, y))
    start_point = stop_point + cycle / n_points
    return str_y


def generate_linear_curve(n_points=10):
    data_list = list(np.linspace(0, 10, n_points))
    str_list = ' '.join(map(str, data_list))
    return str_list


def set_up_EEG():
    def normalize_data(data, min_val=0, max_val=5):
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
        return normalized_data

    global EEG_epoch
    raw = mne.io.read_raw(EEG_path, verbose=False)
    raw.crop(tmax=1000)
    raw.resample(20)
    epochs = mne.make_fixed_length_epochs(raw, duration=0.5, preload=True)
    EEG_epoch = epochs.get_data()[:, 0, :].copy()
    EEG_epoch = normalize_data(EEG_epoch, min_val=0, max_val=15)


def set_up_Serial():
    global ser, serial_success_load
    try:
        ser = serial.Serial(Com_name, 9600)
        serial_success_load = True
    except:
        print(f"Can not open {Com_name}")
    if not ser.is_open:
        print(f"Can not open {Com_name}")


def generate_EEG_curve():
    global EEG_epoch, current_index
    EEG_data = EEG_epoch[current_index]
    current_index = (current_index + 1) % EEG_epoch.shape[0]
    EEG_data = ' '.join(map(str, EEG_data))
    return EEG_data


def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if 'Arduino' in port.description or 'USB' in port.description:
            return port.device
    return None


def set_up():
    set_up_EEG()
    set_redis()
    # set_up_Serial()
    set_UI()
    print("Set up done")


def logic_for_fake_data(operation, stop_event):
    # element = generate_linear_curve()
    # element = generate_sin_curve(n_points=10, y_range=(0, 5))
    # element = generate_EEG_curve()
    element = operation()

    while not stop_event.is_set():
        push_result = push_to_queue(element)
        if push_result == True:
            element = operation()
            # element = generate_linear_curve()
            # element = generate_sin_curve(n_points=10, y_range=(0, 5))
            # element = generate_EEG_curve()
            time.sleep(0.65)
        else:
            time.sleep(1)
    # return None


def logic_for_realtime_data(stop_event):
    global ser, list_data
    set_up_Serial()
    if not serial_success_load:
        messagebox.showinfo("Warning", "Arduino Not Connect, Might not work well")
        return
    try:
        thread = threading.Thread(target=upload_data_batch, args=[stop_event, 10])
        thread.daemon = True
        thread.start()
        while not stop_event.is_set():
            single_data = ser.readline()
            if single_data:
                decoded_data = single_data.decode('utf-8', errors='ignore')
                list_data.append(decoded_data)
    except KeyboardInterrupt:
        print("Error")
    # finally:
    #     ser.close()


def upload_data_batch(stop_event, target_size=10, ):
    global list_data
    while not stop_event.is_set():
        time.sleep(0.65)
        upload_datas = np.array(list_data)
        list_data = []
        if len(upload_datas) == 0:
            continue
        upload_datas = resample(upload_datas, target_size)
        element = ' '.join(map(str, upload_datas))
        push_to_queue(element)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    set_up()
    root_windows.mainloop()

    # logic_for_fake_data()
    # logic_for_realtime_data()
    # root_windows.mainloop()
