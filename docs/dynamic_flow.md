## Description: how to remove dynamic flow
The quantization implementation is relied on the `torch.fx` module to trace and manupulate the model, which requires the model to be static. However, there are several cases that the model is not static, which will cause an error. Here we list several cases.
- dynamic shape for the network operator:

We take the cellpose model as an example. We take the code snippet from [here](https://github.com/MouseLand/cellpose/blob/6efb8f85abbfea63b22eea195c4d046f310cbde6/cellpose/resnet_torch.py#L147)
```python
class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        #style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2],x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style
```
During the symbolic tracing step of the `torch.fx` module, the kernel size of the `F.avg_pool2d` is not a constant and is dependant on the input, which will cause an error. This value is determined by the roi size of the input image and the depth of the network. In this case, we fix this number to 28.

- dynamic control conditions:
```python
def func_to_trace(x):
    if x.sum() > 0:
        return torch.relu(x)
    else:
        return torch.neg(x)

```
In this case, the network structure may depends on the input value, which should be avoided.