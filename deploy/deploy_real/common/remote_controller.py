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
        self.lx = data[0]
        self.ly = -data[1]
        self.rx = -data[2]
        self.ry = data[3]
        self.lt = data[4]
        self.rt = data[5]
        self.xx = data[6]
        self.yy = data[7]
        # print(f"lx: {self.lx}, ly: {self.ly}, rx: {self.rx}, ry: {self.ry}, lt: {self.lt}, rt: {self.rt}, xx: {self.xx}, yy: {self.yy}, buttons: {self.button}")