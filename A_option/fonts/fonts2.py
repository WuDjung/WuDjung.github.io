import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
# 创建一些示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)
# 创建一个折线图
plt.plot(x, y)
plt.title("你好")  # 
# 显示图像
plt.show()