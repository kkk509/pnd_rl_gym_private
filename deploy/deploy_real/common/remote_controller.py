import struct


class KeyMap:
    A = 0
    B = 1
    X = 2
    Y = 3
    LB = 4
    RB = 5
    select = 6
    start = 7
    home = 8
    lo = 9
    ro = 10



class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.lt = 0
        self.rt = 0
        self.xx = 0
        self.yy = 0
        self.button = [0] * 10

    def set(self, data):
        # wireless_remote
        for i in range(10):
            self.button[i] = data [i + 8]
        # for i in range():
        #     self.button[i] = data[27 + i]