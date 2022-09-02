import torch
import _C as P
import time


# x = torch.randn(128,4096)
# x_1,x_2,x_3,x_4 = x.chunk(4,1)

x = torch.load("/localdisk/yejinglai/work/rnnt/gt_fp32.pt")

x_16 = x.to(torch.float16)
start = time.perf_counter()
for i in range(1000):   
    y_a16 = P.tanh_f16(x_16)
end = time.perf_counter()
y = torch.tanh(x)
y_diff = abs(y-y_a16)
print(y_a16)
print(y_diff.max())
print('time cost:%s ms' % ((end - start)*1000))

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
