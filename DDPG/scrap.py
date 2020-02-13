import copy

class A():
    def __init__(self):
        self.a = 0

class B():
    def __init__(self):
        self.b = 1


inst = A()
inst.a
inst.b = B()
inst.b.b

inst.b = B
inst.b.b
inst.b().b

inst.a3
inst.b()
inst.a

inst2 = inst

inst2.a
inst.a
inst2.a = 1
inst2.a
inst.a

inst == inst2
inst is inst2

inst3 = copy.deepcopy(inst)
inst3.a
inst.a
inst2.a
inst2.a=5
inst.a
inst2.a
inst3.a

import numpy as np
A=np.arange(1,5)
np.multiply(A)
help(np.multiply)
np.prod(A)
