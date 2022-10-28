import numpy as np
"""
简单地求二维平面上点到直线的投影点
已知条件：直线外一点，直线上两点坐标
"""

def projection(p,a,b):
    v = np.array([b[0]-a[0],b[1]-a[1]])
    u = np.array([p[0]-a[0],p[1]-a[1]])
    return a+v*(np.dot(v,u)/np.dot(v,v))

if __name__=="__main__":
    p = np.array([2,4])
    a = np.array([0,1])
    b = np.array([1,2])
    res = projection(p,a,b)
    print(res)