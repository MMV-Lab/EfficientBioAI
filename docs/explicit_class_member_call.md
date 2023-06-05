## Avoid explicit Invocation of Custom Class Members Outside the Class
This restriction also comes from the implementation of `torch.fx`(to do the quantization). `torch.fx` will translate our `torch.nn.Module` into `torch.fx.GraphModule`, which is a `torch.nn.Module` instance that holds a Graph as well as a forward method generated from the Graph. During this process, all the self-defined class members will be discarded. Two cases is listed below:

- The first is also from the cellpose repo. The code snippet is [here](https://github.com/MouseLand/cellpose/blob/6efb8f85abbfea63b22eea195c4d046f310cbde6/cellpose/models.py#L369).
```python
if self.pretrained_model:
   self.net.load_model(self.pretrained_model[0], device=self.device)
```
In this case, the `self.net` is a `torch.nn.Module` instance, and is the target for our quantization function. During the calibration process in the quantization, the `self.net` will be transformed to the `torch.fx.GraphModule` and will cause an error when the `self.net.load_model` is called.

- The second cases come from the Deconoising repo. The code snippet is from [here](https://github.com/juglab/DecoNoising/blob/780acecb4603cc12495c0b72351210ff8c098a13/deconoising/prediction.py#L32)
```python
def predict(im, net, device, outScaling):
    stdTorch=torch.Tensor(np.array(net.std)).to(device)
    meanTorch=torch.Tensor(np.array(net.mean)).to(device)
    ...
```
In this example, we use this function as our calibration function. However, the `net` is a `torch.nn.Module` instance, which will be transformed to `torch.fx.GraphModule` and cause an error when the `net.std` is called. We can precomputed these values.