import numpy as np



"""
e-greedy法的简单实现
"""


def epsilon_greedy(epsilon, action_dim, q_values):
    if 1 - epsilon > np.random.uniform(0, 1):
        # 选择Q(s,a)最大对应的动作
        action = np.argmax(q_values)
    else:
        # 随机选择动作
        action = np.random.choice(action_dim)
    return action
