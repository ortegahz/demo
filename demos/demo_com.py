import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
import random


def plot(db, keys, colors, pause_time_s=1):
    plt.ion()
    # for key in keys:
    for i, key in enumerate(keys):
        time_idxs = range(len(db[key]))
        plt.subplot(len(keys), 1, i + 1)
        plt.plot(np.array(time_idxs), np.array(db[key]).astype(float), label=key, color=colors[i])
        plt.legend()
        plt.grid()
    plt.show()
    plt.pause(pause_time_s)
    plt.clf()


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd_in', default=b'\x01\x03\x00\x00\x00\x04\x44\x09')
    parser.add_argument('--db_keys', default=['a', 'b', 'c', 'd'])
    parser.add_argument('--dev_ser', default='/dev/ttyUSB0')
    parser.add_argument('--baud_rate', default=9600)
    parser.add_argument('--interval', default=1)  # second
    return parser.parse_args()


def run(**args):
    colors = list()
    for i in range(len(args['db_keys'])):
        colors.append([random.random(), random.random(), random.random()])
    ser = serial.Serial(args['dev_ser'], args['baud_rate'])
    ser.flushInput()
    db = dict()
    for key in args['db_keys']:
        db[key] = list()
    while True:
        ser.write(args['cmd_in'])
        while ser.inWaiting() < 13:
            continue
        buff_lst = list()
        head_0 = ser.read(1).hex()
        head_1 = ser.read(1).hex()
        head_2 = ser.read(1).hex()
        assert head_0 == '01' and head_1 == '03' and head_2 == '08'
        buff_lst.append(head_0)
        buff_lst.append(head_1)
        buff_lst.append(head_2)

        for _ in range(10):
            buff_lst.append(ser.read(1).hex())
        logging.info(buff_lst)
        data = list()
        data.append(int(buff_lst[3] + buff_lst[4], 16))
        data.append(int(buff_lst[5] + buff_lst[6], 16))
        data.append(int(buff_lst[7] + buff_lst[8], 16))
        data.append(int(buff_lst[9] + buff_lst[10], 16))
        for i, key in enumerate(args['db_keys']):
            db[key].append(data[i])
            # logging.info(db[key][-1])
        plot(db, args['db_keys'], colors)
        time.sleep(args['interval'])


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(**vars(args))


if __name__ == '__main__':
    main()
