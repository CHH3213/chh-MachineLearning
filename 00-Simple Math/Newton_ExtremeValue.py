from re import X
import numpy as np

"""牛顿下降法求函数极值，对于牛顿法而言就是求函数导数的根
"""

def func(x):
    """方程"""
    return -x**2+4*x

def func_prime(x):
    delta = 1e-4
    return (func(x+delta)-func(x))/(delta) 

def func_2prime(x):
    delta = 1e-4
    return (func_prime(x+delta)-func_prime(x))/(delta) 

def Newton(x0,presision):
    """牛顿迭代法

    Args:
        x0 (_type_): 初始值
        learning_rate (_type_): 学习率
        presision (_type_): 精度
    """
    x = x0
    x_last = 0.1
    while(abs(x-x_last) > presision):
        x_last = x
        x = x_last-func_prime(x_last)/func_2prime(x_last)
    print("极值点：", x)
    return x

if __name__=='__main__':
    x = Newton(1,1e-10)
    print("极值：", func(x))

