import numpy as np

"""牛顿下降法求一元方程的根：np.log(x)+x**2=0
"""

def func(x):
    """方程"""
    return np.log(x)+x**2

def func_prime(x):
    delta = 1e-4
    return (func(x+delta)-func(x))/(delta) 



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
        x = x_last-func(x_last)/func_prime(x_last)
        print(x)
    return x

if __name__=='__main__':
    x = Newton(0.5,1e-5)
    print(func(x))

