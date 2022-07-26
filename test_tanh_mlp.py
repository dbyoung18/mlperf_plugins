import torch
import _C as P

# x = torch.load("/localdisk/yejinglai/work/rnnt/gates_fp32.pt")
# x = x.contiguous()
# x_fp32 = x.to(torch.float32)
# x_fp32 = x_fp32.contiguous()

x = torch.randn(128,4096)
x_1,x_2,x_3,x_4 = x.chunk(4,1)

y = torch.sigmoid(x_1)
y_a16 = P.nc_sigmoid(x_1)
y_diff = abs(y-y_a16)
print(y_diff.max())

# x = torch.linspace(-10,10,131072)
# # x = torch.randn(128,1024)
# x = x.reshape(128,1024)
# # x = x.unsqueeze(0)
# x_fp16 = x.to(torch.float16)
# y = torch.tanh(x)

# # y_a32 = P.tanh(x)
# y_a16 = P.tanh(x)
# # y_t = torch.tanh(x)
# y_diff = abs(y-y_a16)
# print(y_diff.max())
