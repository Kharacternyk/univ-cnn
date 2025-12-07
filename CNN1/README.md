```
$ python my_conv.py
torch.Size([2, 2, 1, 1]) torch.Size([2, 2, 1, 1])
tensor([[[[-1.1681]],

         [[-0.1085]]],


        [[[-4.9009]],

         [[ 2.8136]]]], grad_fn=<MyConv2dFunctionBackward>)
tensor([[[[-1.1681]],

         [[-0.1085]]],


        [[[-4.9009]],

         [[ 2.8136]]]], grad_fn=<ConvolutionBackward0>)
Forward check: True
w grad close: True
b grad close: True
```
