import torch
import _C as P
import time
import numpy as np

torch.set_printoptions(6)
gates = torch.load('/localdisk/lyj/work/rnnt-inference/gates.pt')
it, ft, gt, ot = gates.chunk(4, 1)
# it = it.contiguous()
# ft = ft.contiguous()
# gt = gt.contiguous()
# ot = ot.contiguous()
ct_1 = torch.zeros(128,1024).to(torch.float16)
ct = torch.load('/localdisk/lyj/work/rnnt-inference/ct.pt')

ht = P.sigmoid(ot) * P.tanh_f16(ct)
yt = torch.load('/localdisk/lyj/work/rnnt-inference/yt.pt')
quant_ht = torch.load('/localdisk/lyj/work/rnnt-inference/quant_ht.pt')
ya = P.lstm_postop(it,ft,gt,ot,ct_1,torch.tensor(7.0943).item(),127,True)
print((ya[3].eq(ct)==1).all())
print(ya[0])
print(ht)
# print(ht[127][:32])


# x = torch.load("/localdisk/yejinglai/work/rnnt/gt_fp32.pt")

# x_16 = x.to(torch.float16)
# print(x_16.shape)
# start = time.perf_counter()
# for i in range(1000):   
#     y_a16 = P.tanh_f16(x_16)
# end = time.perf_counter()
# y = torch.tanh(x)
# y_diff = abs(y-y_a16)
# print(y_a16)
# print(y_diff.max())
# print('time cost:%s ms' % ((end - start)*1000))

# y = torch.tanh(x_1)
# start = time.perf_counter()
# for i in range(1000):   
#     y_a16 = torch.tanh(x)
# end = time.perf_counter()
# print('appro time cost:%s ms' % ((end - start)*1000))

# start = time.perf_counter()
# y = torch.tanh(x)
# end = time.perf_counter()
# print('toch time cost:%s ms' % ((end - start)*1000))


# y_diff = abs(y-y_a16)
# print(y_diff.max())
