import numpy as np
# 运行两套网格后
V1 = 0.020507  # S=K 处结果
V2 = 0.013987
p  = 2                 # CN 二阶
V_ext = V2 + (V2 - V1)/(2**p - 1)
err_est = abs(V2 - V1)/(2**p - 1)
print('外推解 V_ext = {:.7f}'.format(V_ext))
print('误差估计     = {:.2e}'.format(err_est))

# 若再跑第三套 M=1600,N=8000 得 V3
V3 =  0.010740
EOC = np.log(abs(V2-V1)/abs(V3-V2))/np.log(2)
print('实验收敛阶 EOC = {:.3f}'.format(EOC))