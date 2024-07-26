import re
import matplotlib.pyplot as plt

def parse_loss_from_log(file_path):
    """解析日志文件中的损失值"""
    iter_nums = []
    loss_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'iter (\d+): loss ([\d.]+)', line)
            if match:
                iter_nums.append(int(match.group(1)))
                loss_values.append(float(match.group(2)))
    return iter_nums, loss_values

def plot_loss_curves(files):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    for file_path in files:
        iter_nums, loss_values = parse_loss_from_log(file_path)
        N = 2400
        plt.plot(iter_nums[:N], loss_values[:N], label='.'.join(file_path.split('.')[:-1]))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.savefig('train_loss.png')

plot_loss_curves([
    'nanoLLaMA(33M)-Jittor1.3.9.10-Nvidia.log',
    'nanoLLaMA(33M)-Jittor1.3.9.10-Huawei.log'
])
