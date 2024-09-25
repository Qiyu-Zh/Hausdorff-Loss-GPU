# Hausdorff-Loss-GPU

In this package, you can calculate hausdorff distance loss in GPU easily.
Exampe:
```
x = torch.rand(2,10,32,32).cuda()
y = torch.randint(0, 10, (2, 1, 32, 32)).cuda()
loss = HD_loss(apply_nonlin=None)
res = loss(x, y)
print(res)
```

```
import torch.nn as nn
softmax = nn.Softmax(dim=1)
x = torch.rand(2, 10, 32, 32, 32).cuda()
y = torch.randint(0, 10, (2, 1, 32, 32, 32)).cuda()
loss = HD_loss(apply_nonlin=softmax, alpha = 2)
res = loss(x, y)
print(res)
```

You can add parameter for alpha. If you do this, the function will choose the power for Hausdorff distance:

You can install it through:
```
!pip install git+https://github.com/Qiyu-Zh/Hausdorff-Loss-GPU.git
```
