from re import X
import numpy as np

"""梯度下降法求一元方程的根：np.log(x)+x**2=0
"""

def func(x):
    """方程"""
    return np.log(x)+x**2


def loss_func(x):
    """损失函数即方程: func(x)=0"""
    return (func(x)-0)**2
def loss_func_prime(x):
    """导数"""
    # 使用导数定义方式求导数
    delta = 1e-4
    return (loss_func(x+delta)-loss_func(x))/(delta) 

def GradientDescent(x0,learning_rate,presision):
    """梯度下降

    Args:
        x0 (_type_): 初始值
        learning_rate (_type_): 学习率
        presision (_type_): 精度
    """
    x = x0
    while(abs(loss_func(x)) > presision):
        x = x-learning_rate*loss_func_prime(x)
    print(x)
    return x

if __name__=='__main__':
    x = GradientDescent(1,0.01,1e-10)
    print(func(x))

