import torch
import numpy as np


def autograd_demo():
    x = torch.ones(5)
    y = torch.randn(5, 5, requires_grad=True)
    b = torch.randn_like(x)
#     grad_list = []

    # def aa(grad):
    #     grad_list.append(grad)
    for i in range(100):
        # if i > 0:
        #     y.grad.zero_()
        # y.register_hook(aa)
        # y.retain_grad()
        z = torch.matmul(y, x) + b
        output = torch.sigmoid(z)
        label = torch.tensor([0, 0, 1, 0, 0])
        loss = (output - label).var()
        loss.backward()
        # if i < 90:
        #     y = y - 0.2 * y.grad
        # else:
        #     y = y - 0.05 * y.grad
        # y = y - 0.2 * grad_list[-1]
        # y = y - 0.2 * y.grad
        with torch.no_grad():
            # v = y.view(5, 5)
            # v.sub_(0.1 * y.grad)
            y.sub_(0.2 * y.grad)
        # y = torch.01tensor(y, requires_grad=True)
        print(loss)


# def internal_grad_demo():
#     x = torch.ones(5)
#     y = torch.zeros(3)
#     w = torch.randn(5, 3, requires_grad=True)
#     b = torch.randn(3, requires_grad=True)
#     z = torch.matmul(x, w) + b
#     o = z.sigmoid()
#     print(z.shape)
#     out_puts = torch.ones_like(z)
#     o.backward(out_puts)
#     print(w.grad)


if __name__ == "__main__":
    autograd_demo()
    # internal_grad_demo()
    print("run autograd_demo.py successful")
