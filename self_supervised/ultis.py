from itertools import permutations
import numpy as np

class PN_converter():
    def __init__(self,length, pnumber=0):
        if pnumber==0:
            self.n2p = list(permutations(np.array(range(int(length)))))
        else:
            self.n2p = []
            while len(self.n2p)<pnumber:
                t=tuple(np.random.permutation(length))
                if t not in self.n2p:
                    self.n2p.append(t)
        self.p2n = {}
        for i in range(len(self.n2p)):
            self.p2n[self.n2p[i]] = i

    def num2perm(self, num):
        return self.n2p[num]

    def perm2num(self, perm):
        perm=tuple(perm)
        return self.p2n[perm]

# pnc=PN_converter(9, 10)
# print(pnc.n2p)
# print(pnc.p2n)




