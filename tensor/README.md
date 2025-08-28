# pytorch Tensor guide

## tensor是什么
- 张量 (weight activation)
- 多维数据 numpy --> ndarray
- pytorch里面最大的一个类

## 初始化一个tensor
- torch.tensor() 调用tensor这个类的init方法完成初始化
- torch.ones() 用torch本身自带的一些函数来生成特殊类型的tensor
- torch.ones_like() 也是调用torch本身自带函数，shape采用like括号里面的shape
- numpy与tensor之间互相转换: 优先选用from_numpy()

## tensor 的构成
1. meta_ data
2. raw_data

## to设备常见写法
(‘’‘)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
(’‘’)