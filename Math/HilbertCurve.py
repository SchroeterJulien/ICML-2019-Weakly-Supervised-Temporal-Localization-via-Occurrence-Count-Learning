# File containing the function softmax which allows to compute the softmax of an array

import numpy as np


def HilbertCurve(n):
    ''' Generate Hilbert curve indexing for (n, n) array. 'n' must be a power of two. '''
    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = HilbertCurve(n // 2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size * 2
    d = np.flipud(np.rot90(t, -1)) + t.size * 3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))


def SnailCurve(n):
    x = [-1]
    y = [0]
    direction = 0  # left, up, right, dowm
    extent = n
    while extent > 0:
        for ii in range(extent):
            if direction == 0:
                x.append(x[-1] + 1)
                y.append(y[-1])
            elif direction == 1:
                x.append(x[-1])
                y.append(y[-1] + 1)
            elif direction == 2:
                x.append(x[-1] - 1)
                y.append(y[-1])
            else:
                x.append(x[-1])
                y.append(y[-1] - 1)

            if ii == extent -1:
                direction = (direction + 1) % 4

                if direction in [1,3]:
                    extent -= 1

    x.pop(0), y.pop(0)
    return np.array(x), np.array(y)


def ScanCurve(n):
    x = [-1]
    y = [-1]
    for ii in range(n):
        for kk in range(n):

            if kk == 0:
                x.append(x[-1])
                y.append(y[-1] + 1)
            else:
                if ii % 2 == 0:
                    x.append(x[-1] + 1)
                    y.append(y[-1])
                else:  # turn back
                    x.append(x[-1] - 1)
                    y.append(y[-1])

    x.pop(0), y.pop(0)
    return np.array(x)+1, np.array(y)


def MooreCurve(n):
    iter = int(np.sqrt(n)- 1)

    # L-system
    axiom = 'LFL+F+LFL'

    for ii in range(iter):
        axiom = axiom.replace('L','l').replace('R','r')
        axiom = axiom.replace('l','−RF+LFL+FR−')
        axiom = axiom.replace('r','+LF−RFR−FL+')

    axiom = axiom.replace('R','')
    axiom = axiom.replace('L','')

    # Move
    x = [0]
    y = [0]
    direction = 0
    for action in axiom:
        if action == 'F':
            if direction == 0:
                x.append(x[-1] + 1)
                y.append(y[-1])
            elif direction == 1:
                x.append(x[-1])
                y.append(y[-1] + 1)
            elif direction == 2:
                x.append(x[-1] - 1)
                y.append(y[-1])
            else:
                x.append(x[-1])
                y.append(y[-1] - 1)
        elif action == '+':
            direction = (direction + 1) % 4
        elif action == '−':
            direction = (direction - 1) % 4
    
    return np.array(x), np.array(y)+n/2-1

