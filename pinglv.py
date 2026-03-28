import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#网络定义
class PINN(nn.Module):
    """普通全连接网络"""

    def __init__(self, layers=[1, 50, 50, 1]):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear{i}', nn.Linear(layers[i], layers[i + 1]))
            self.net.add_module(f'tanh{i}', nn.Tanh())
        self.net.add_module('output', nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        return self.net(x)


class FourierFeatureNetwork(nn.Module):
    """Fourier Feature 网络"""

    def __init__(self, input_dim=1, hidden_dims=[20, 20], output_dim=1, num_frequencies=20):
        super().__init__()
        # 随机频率矩阵
        self.B = torch.randn(input_dim, num_frequencies) * 2.0  # 可调整尺度
        self.net = nn.Sequential(
            nn.Linear(2 * num_frequencies, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.B  # (batch, num_freq)
        features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=1)
        return self.net(features)


class MultiScaleFourierNetwork(nn.Module):
    """多尺度 Fourier Feature 网络"""

    def __init__(self, input_dim=1, hidden_dims=[50, 50], output_dim=1, num_scales=3, num_freq_per_scale=10):
        super().__init__()
        self.scales = []
        total_features = 0
        for i in range(num_scales):
            B = torch.randn(input_dim, num_freq_per_scale) * (2.0 ** i)  # 尺度递增
            self.register_buffer(f'B_{i}', B)
            self.scales.append(B)
            total_features += 2 * num_freq_per_scale
        self.net = nn.Sequential(
            nn.Linear(total_features, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        features = []
        for B in self.scales:
            x_proj = 2 * torch.pi * x @ B
            features.append(torch.cos(x_proj))
            features.append(torch.sin(x_proj))
        features = torch.cat(features, dim=1)
        return self.net(features)


#训练函数
def train_pinn(model, f_func, epochs=1000, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    for ep in range(epochs):
        # 内部点（随机采样）
        x = torch.rand(1000, 1, requires_grad=True, device=device)
        u = model(x)
        # 一阶导
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # 二阶导
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        # 源项
        f_val = f_func(x)
        loss_pde = criterion(u_xx + f_val, torch.zeros_like(u_xx))

        # 边界点
        x0 = torch.tensor([[0.0]], requires_grad=True, device=device)
        x1 = torch.tensor([[1.0]], requires_grad=True, device=device)
        u0 = model(x0)
        u1 = model(x1)
        loss_bc = criterion(u0, torch.zeros_like(u0)) + criterion(u1, torch.zeros_like(u1))

        loss = loss_pde + loss_bc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if ep % 1000 == 0:
            print(f'Epoch {ep}, Loss = {loss.item():.2e}')
    return model, loss_history


#解析解
def exact_solution_single_freq(x):
    """单一频率 f=sin(2πx) 的解析解 u = sin(2πx)/(4π²)"""
    return torch.sin(2 * np.pi * x) / (4 * np.pi ** 2)


def exact_solution_multi_freq(x):
    """多频率 f=sin(2πx)+0.5sin(10πx) 的解析解"""
    return (torch.sin(2 * np.pi * x) / (4 * np.pi ** 2) +
            0.5 * torch.sin(10 * np.pi * x) / (100 * np.pi ** 2))


#绘图
def plot_comparison(model, x_test, exact_func, title):
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy()
    u_exact = exact_func(x_test).cpu().numpy()
    plt.figure()
    plt.plot(x_test.cpu().numpy(), u_pred, 'b-', label='PINN')
    plt.plot(x_test.cpu().numpy(), u_exact, 'r--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(title)
    plt.legend()
    plt.show()
    error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f'Relative error: {error:.2e}')


#主程序
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)


    #定义源项
    def f_single(x): return torch.sin(2 * np.pi * x)


    def f_multi(x): return torch.sin(2 * np.pi * x) + 0.5 * torch.sin(10 * np.pi * x)


    #单一频率测试
    print("=== Single Frequency ===")
    model_nn = PINN().to(device)
    model_nn, loss_nn = train_pinn(model_nn, f_single, epochs=1000)
    plot_comparison(model_nn, x_test, exact_solution_single_freq, 'NN')

    model_ff = FourierFeatureNetwork().to(device)
    model_ff, loss_ff = train_pinn(model_ff, f_single, epochs=1000)
    plot_comparison(model_ff, x_test, exact_solution_single_freq, 'Fourier Feature')

    model_mff = MultiScaleFourierNetwork().to(device)
    model_mff, loss_mff = train_pinn(model_mff, f_single, epochs=1000)
    plot_comparison(model_mff, x_test, exact_solution_single_freq, 'Multi-scale Fourier Feature')

    #多频率测试
    print("=== Multi Frequency ===")
    model_nn_multi = PINN().to(device)
    model_nn_multi, _ = train_pinn(model_nn_multi, f_multi, epochs=1000)
    plot_comparison(model_nn_multi, x_test, exact_solution_multi_freq, 'NN (Multi-frequency)')

    model_ff_multi = FourierFeatureNetwork().to(device)
    model_ff_multi, _ = train_pinn(model_ff_multi, f_multi, epochs=1000)
    plot_comparison(model_ff_multi, x_test, exact_solution_multi_freq, 'Fourier Feature (Multi-frequency)')

    model_mff_multi = MultiScaleFourierNetwork().to(device)
    model_mff_multi, _ = train_pinn(model_mff_multi, f_multi, epochs=1000)
    plot_comparison(model_mff_multi, x_test, exact_solution_multi_freq, 'Multi-scale Fourier Feature (Multi-frequency)')