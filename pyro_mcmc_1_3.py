"""
曲奇饼问题的变分推断解法
曲奇饼问题：假设有两碗曲奇饼，碗1包含30个香草曲奇饼和10个巧克力味曲奇饼，碗2各有上述两种饼干各20个。
问题1：随机挑了一个碗有放回地拿出1块曲奇饼，得到了1块香草味曲奇饼，那么这个曲奇饼是从碗1中拿出的概率是多少？
问题2：随机挑了一个碗有放回地拿出5块曲奇饼，得到了3块香草味曲奇饼和2块巧克力味曲奇饼，那么这些曲奇饼是从碗1中拿出的概率是多少？
问题3：随机挑了一个碗有放回地拿出5块曲奇饼，得到曲奇饼味道的顺序为[V, C, V, C, V]，其中V表示香草味，C表示巧克力味，那么这些曲奇饼是从碗1中拿出的概率是多少？
"""

import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from scipy.special import comb

# 设置随机数种子
pyro.set_rng_seed(101)


############################
### 1.建立包含随机变量的模型 ###
###########################

# 1.1.问题1的模型
def cookie1():
    # 由于只有两个碗，因此随机拿一个碗的概率是1/2，即是否拿到碗1的先验分布服从Bernoulli(0.5)
    bowl1 = pyro.sample('bowl1', pyro.distributions.Uniform(0.0, 1.0))
    bowl = 'bowl1' if bowl1.item() >= 0.5 else 'bowl2'
    # 由于每个碗里面不是香草味就是巧克力味，因此选好碗之后有放回地取饼干中的香草味饼干个数服从二项分布
    ratio = {'bowl1': 30.0 / (30.0 + 10.0), 'bowl2': 20.0 / (20.0 + 20.0)}[bowl]  # 每个碗中香草味曲奇饼占比
    vanilla = pyro.sample('vanilla', pyro.distributions.Bernoulli(ratio))  # 只取了1次，显然是伯努利分布
    return vanilla


# 1.2.问题2的模型
def cookie2(n):  # n是取曲奇饼的次数
    # 由于只有两个碗，因此随机拿一个碗的概率是1/2，即是否拿到碗1的先验分布服从Bernoulli(0.5)
    bowl1 = pyro.sample('bowl1', pyro.distributions.Uniform(0.0, 1.0))
    bowl = 'bowl1' if bowl1.item() >= 0.5 else 'bowl2'
    # 由于每个碗里面不是香草味就是巧克力味，因此选好碗之后有放回地取饼干服从二项分布
    ratio = {'bowl1': 30.0 / (30.0 + 10.0), 'bowl2': 20.0 / (20.0 + 20.0)}[bowl]  # 每个碗中香草味曲奇饼占比
    vanillas = pyro.sample('vanillas', pyro.distributions.Binomial(n, ratio))  # 取了n次
    return vanillas


# 1.3.问题3的模型
def cookie3(n):  # n是取曲奇饼的次数
    # 由于只有两个碗，因此随机拿一个碗的概率是1/2，即是否拿到碗1的先验分布服从Bernoulli(0.5)
    bowl1 = pyro.sample('bowl1', pyro.distributions.Uniform(0.0, 1.0))
    bowl = 'bowl1' if bowl1.item() >= 0.5 else 'bowl2'
    # 由于每个碗里面不是香草味就是巧克力味，因此选好碗之后有放回地取饼干服从二项分布
    ratio = {'bowl1': 30.0 / (30.0 + 10.0), 'bowl2': 20.0 / (20.0 + 20.0)}[bowl]  # 每个碗中香草味曲奇饼占比
    # 取了n次
    vanillas = []
    for k in range(n):
        vanillas.append(pyro.sample('vanilla_{}'.format(k), pyro.distributions.Bernoulli(ratio)))
    return vanillas


################################
### 2.使用观测值作为模型的约束条件 ###
################################

# 2.1.问题1的观测值作为问题1的约束条件
conditioned_cookie1 = pyro.condition(cookie1, data={"vanilla": torch.tensor(1).float()})

# 2.2.问题2的观测值作为问题2的约束条件
conditioned_cookie2 = pyro.condition(cookie2, data={"vanillas": torch.tensor(3).float()})

# 2.3.问题3的观测值作为问题3的约束条件
vanillas = [1, 0, 1, 0, 1]
data = {}
for k in range(5):
    data["vanilla_{}".format(k)] = torch.tensor(vanillas[k]).float()
# conditioned_cookie3 = pyro.condition(cookie3, data={"vanilla_0": torch.tensor([1]).float(), "vanilla_1": torch.tensor([0]).float(), "vanilla_2": torch.tensor([1]).float(), "vanilla_3": torch.tensor([0]).float(), "vanilla_4": torch.tensor([1]).float()})
conditioned_cookie3 = pyro.condition(cookie3, data=data)


################################
##### 3.进行MCMC，确定参数值 #######
################################

# 3.1.求解问题1的程序
def cookie1_solutions():
    nuts_kernel = NUTS(conditioned_cookie1, jit_compile=False)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=10000,
        warmup_steps=10000,
        num_chains=1,
    )
    mcmc.run()
    samples = mcmc.get_samples()['bowl1']
    # 以0.5为阈值，将连续转换为离散
    bowl1 = torch.where(samples >= 0.5, 1.0, 0.0)
    print("Computional value:", torch.mean(bowl1).item())
    print("Theoretical value:", 0.6)
    # 绘图
    samples = samples.detach().cpu().numpy()
    plt.hist(samples, bins=100)
    plt.show()


# 3.2.求解问题2的程序
def cookie2_solutions():
    nuts_kernel = NUTS(conditioned_cookie2, jit_compile=False)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=10000,
        warmup_steps=10000,
        num_chains=1,
    )
    mcmc.run(5)
    samples = mcmc.get_samples()['bowl1']
    bowl1 = torch.where(samples >= 0.5, 1.0, 0.0)
    print("Computional value:", torch.mean(bowl1).item())
    print("Theoretical value:",
          (comb(5, 3) * 0.75 ** 3 * 0.25 ** 2) / (comb(5, 3) * 0.75 ** 3 * 0.25 ** 2 + comb(5, 3) * 0.5 ** 5))
    # 绘图
    samples = samples.detach().cpu().numpy()
    plt.hist(samples, bins=100)
    plt.show()


def cookie3_solutions():
    nuts_kernel = NUTS(conditioned_cookie3, jit_compile=False)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=10000,
        warmup_steps=10000,
        num_chains=1,
    )
    mcmc.run(5)
    samples = mcmc.get_samples()['bowl1']
    bowl1 = torch.where(samples >= 0.5, 1.0, 0.0)
    print("Computional value:", torch.mean(bowl1).item())
    print("Theoretical value:", (0.75 ** 3 * 0.25 ** 2) / (0.75 ** 3 * 0.25 ** 2 + 0.5 ** 5))
    # 绘图
    samples = samples.detach().cpu().numpy()
    plt.hist(samples, bins=100)
    plt.show()


# cookie1_solutions()
# cookie2_solutions()
cookie3_solutions()
