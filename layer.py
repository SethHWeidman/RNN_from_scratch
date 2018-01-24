from activation import Sigmoid, Tanh
from gate import AddGate, MultiplyGate
import numpy as np

mulGate = MultiplyGate()
addGate = AddGate()
sigmoid = Sigmoid()
tanh = Tanh()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)


class LSTMLayer:
    def forward(self, x, c_prev, h_prev, Wf, Wi, Wc, Wo, Wv):
        self.z = np.rowstack((h_prev, x))
        self.f_in = mulGate.forward(Wf, self.z)
        self.f_out = sigmoid.forward(self.f_in)
        self.i_in = mulGate.forward(Wi, self.z)
        self.i_out = sigmoid.forward(self.i_in)
        self.c_new_in = mulGate.forward(Wc, self.z)
        self.c_new_out = tanh.forward(self.c_new_in)
        self.c = addGate.forward(self.f_out * c_prev, self.i_out * self.c_new_out)
        self.o_in = mulGate.forward(Wo, self.z)
        self.o_out = sigmoid.forward(self.o_in)
        self.c_tanh = tanh.forward(self.c)
        self.h = mulGate.forward(self.o_out, self.c_tanh)
        self.V = mulGate.forward(Wv, self.h)

    def backward(self, x, c_prev, h_prev, Wf, Wi, Wc, Wo, Wv, c_diff, h_diff, v_diff):
        self.forward(x, c_prev, h_prev, Wf, Wi, Wc, Wo, Wv)
        self.dz = np.zeros_like(self.dz)
        dWv, dh = mulGate.backward(Wv, self.h, v_diff)
        do_out, dc_tan = mulGate.backward(self.o_out, self.c_tanh, h_diff)
        dc = tanh.backward(self.c, dc_tan)
        do_in = sigmoid.backward(self.o_in, do_out)
        dWo, dz_add = mulGate.backward(Wo, self.z, do_in)
        self.dz += dz_add
        df_out = dc * c_prev
        dc_prev = dc * self.f_out
        dc_new_out = dc * self.i_out
        di_out = dc * self.c_new_out
        dc_new_in = tanh.backward(self.c_new_in, dc_new_out)
        dWc, dz_add =  mulGate.backward(Wc, self.z, dc_new_in)
        self.dz += dz_add
        di_in = sigmoid.backward(self.i_in, di_out)
        dWi, dz_add = mulGate.backward(Wi, self.z, di_in)
        self.dz += dz_add
        df_in = sigmoid.backward(self.f_in, df_out)
        dWf, dz_add = mulGate.backward(Wf, self.z, df_in)
        self.dz += dz_add

        dh_prev = self.dz[:H_size, :]
        dx_prev = self.dz[H_size:, :]

        return dh_prev, dc_prev, dx_prev
