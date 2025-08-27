import torch
import numpy as np


# def tensor_creat():
    # 方式1
    # data = [[1, 2], [3, 4]]
    # x_data = torch.tensor(data)
    # x_data2 = torch.tensor((1, 2, 3))
    # print(x_data2)

    # 方式2
    # data = torch.ones(1, 2, 3)
    # data2 = torch.randn(1, 2, 3)
    # print(data2)
    # print(data2.shape)

    # 方式3
    # data = torch.tensor([[1, 2], [3, 4]])
    # data1 = torch.ones_like(data)
    # data2 = torch.empty_like(data)
    # print(data2.shape)
    # print(data2)

    # 方式4
    # arr = np.array([[1, 2], [4, 5]])
    # data = torch.from_numpy(arr)
    # data1 = torch.tensor(arr)
    # arr1 = data.numpy()
    # print(data)
    # print(type(arr1))
    # print(data1)


# def tensor_struct():
    # arr = np.array([[1, 2], [3, 4]])
    # data = torch.tensor(arr)
    # data = torch.from_numpy(arr)

    # meat_data
    # print(data.shape)
    # print(data.stride())
    # print(data.dtype)
    # print(data.device)
    # arr = data.numpy()
    # print(arr.dtype)

    # raw_data
    # print(data)
    # print(arr.ctypes.data)
    # print(data.data_ptr())
    # data2 = data.reshape((1, 4))
    # print(id(data))
    # print(id(data2))
    # print(data.data_ptr())
    # print(data2.data_ptr())
    # print(data)

# def tensor_view():
    # a = torch.arange(24).reshape(3, 8)
    # b = a.T
    # print(a.storage())
    # print(b.storage())
    # print(a.data_ptr())
    # print(b.data_ptr())

# def stride_demo():
    # a = torch.arange(12).reshape(3, 4)
    # b = a.reshape(4, 3)
    # print(a.stride())
    # print(b.stride())
    # a = torch.arange(24).reshape(2, 3, 4)
    # b = a.permute(2, 0, 1).contiguous()
    # c = b.view(4, 6)
    # d = torch.Tensor([1, 2, 3])
    # print(a.stride())
    # print(a.shape)
    # print(b.stride())
    # print(b.shape)
    # print(c.stride())
    # print(c.shape)
    # print(b.data_ptr())
    # print(c.data_ptr())
    # print(a.data_ptr())

def tensor_to_demo():
    data = torch.ones(4, 5)
    data1 = data.to("cuda:0")
    data1 = data1.to(torch.float64)
    # data2 = data.cuda()

    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")
    # data1 = data.to(device)
    # print(data.device)
    # print(data1.device)
    # print(data2.device)
    # print(data.device)
    data2 = data.to(data1)
    print(data1.dtype)
    print(data2.device)
    print(data2.dtype)

if __name__ == "__main__":
    # tensor_creat()
    # tensor_struct()
    # tensor_view()
    # stride_demo()
    tensor_to_demo()
    print("Run tensor_demo.py Successful")