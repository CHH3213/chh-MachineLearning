from re import X
import numpy as np

"""简单的梯度上升法求解函数最值实现
求函数f(x)=-x^2 + 4x的极大值
"""

def func(x):
    return -x**2+4*x
def func_prime(x):
    """导数"""
    delta = 1e-5
    return (func(x+delta)-func(x))/delta
def GradientAscent(x0,learning_rate,presision):
    """梯度上升

    Args:
        x0 (_type_): 初始值
        learning_rate (_type_): 学习率
        presision (_type_): 精度
    """
    x = x0
    x_next = 0
    while(abs(x_next-x)>presision):
        x=x_next
        x_next = x+learning_rate*func_prime(x)
    print("极值点：", x)

if __name__=='__main__':
    GradientAscent(1,0.01,1e-10)

