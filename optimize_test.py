import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tieloss import TIELoss

class TIESolver:
    def __init__(self, image_size=128, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.tieloss = TIELoss(
            image_size=image_size,
            wavelength_m=632.8e-9,
            pixel_size_m=8e-6,
            loss_type='mse'
        ).to(self.device)
    
    def create_test_data(self):
        x = torch.linspace(-1, 1, self.image_size, device=self.device)
        y = torch.linspace(-1, 1, self.image_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        true_phase = (
            torch.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.2) * 2.0
            + torch.exp(-((X + 0.4) ** 2 + (Y + 0.2) ** 2) / 0.15) * 1.5
            + 0.5 * torch.sin(3 * X) * torch.cos(2 * Y)
        )
        true_phase = true_phase / true_phase.abs().max()
        intensity = 0.8 + 0.2 * torch.exp(-(X ** 2 + Y ** 2) / 2.0)
        return true_phase, intensity
    
    def compute_target_dI_dz(self, true_phase, intensity):
        phi_grad_x, phi_grad_y = self.tieloss._gradient(true_phase)
        div = self.tieloss._divergence(
            intensity * phi_grad_x,
            intensity * phi_grad_y
        )
        return -div / self.tieloss.k
    
    def create_initial_guess(self, true_phase, method='gaussian_smooth'):
        if method == 'random':
            mean = true_phase.mean().item()
            std = true_phase.std().item()
            return torch.randn_like(true_phase) * std + mean
        elif method == 'gaussian_smooth':
            from scipy.ndimage import gaussian_filter
            true_np = true_phase.cpu().numpy()
            blurred = gaussian_filter(true_np, sigma=5.0)
            return torch.from_numpy(blurred).to(true_phase.device)
        else:
            return torch.randn_like(true_phase) * 0.1
    
    def optimize_with_boundary(self, true_phase, intensity, target_dI_dz, 
                              boundary_width=5, n_epochs=1000, lr=0.1):
        internal_mask = torch.ones_like(true_phase, dtype=torch.bool)
        internal_mask[:boundary_width, :] = False
        internal_mask[-boundary_width:, :] = False
        internal_mask[:, :boundary_width] = False
        internal_mask[:, -boundary_width:] = False
        internal_phase = nn.Parameter(
            true_phase[internal_mask].clone().detach()
        )
        def forward():
            full_phase = true_phase.clone()
            full_phase[internal_mask] = internal_phase
            return full_phase
        optimizer = optim.AdamW([internal_phase], lr=lr)
        for epoch in range(n_epochs):
            current_phase = forward()
            loss = self.tieloss(current_phase, intensity, target_dI_dz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch > 100 and loss.item() < 1e-8:
                break
        final_phase = forward().detach()
        return final_phase
    
    def optimize_without_boundary(self, true_phase, intensity, target_dI_dz,
                                 init_method='gaussian_smooth',
                                 n_epochs=2000, lr=0.05):
        phase_param = nn.Parameter(
            self.create_initial_guess(true_phase, method=init_method)
        )
        optimizer = optim.AdamW([phase_param], lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        for epoch in range(n_epochs):
            current_phase = phase_param
            loss = self.tieloss(current_phase, intensity, target_dI_dz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch > 500 and loss.item() < 1e-9:
                break
        final_phase = phase_param.detach()
        return final_phase
    
    def analyze_result(self, pred_phase, true_phase, description):
        aligned = pred_phase - pred_phase.mean() + true_phase.mean()
        mse = float(torch.mean((aligned - true_phase) ** 2).item())
        mae = float(torch.mean(torch.abs(aligned - true_phase)).item())
        pred_flat = aligned.flatten().cpu().numpy()
        true_flat = true_phase.flatten().cpu().numpy()
        corr = float(np.corrcoef(pred_flat, true_flat)[0, 1])
        print(f"\n{description}:")
        print(f"  MSE: {mse:.6e}")
        print(f"  MAE: {mae:.6e}")
        print(f"  相关系数: {corr:.4f}\n")
        return aligned, mse, corr
    
    def plot_results(self, true_phase, results):
        _, axes = plt.subplots(3, 3, figsize=(12, 12))
        phases = {
            '有边界条件': results['phase_with_boundary'],
            '无边界条件（好初始猜测）': results['phase_no_boundary_good'],
            '无边界条件（随机初始猜测）': results['phase_no_boundary_random']
        }
        titles = ['真实相位', '重建相位', '相位误差']
        for row, (key, phase) in enumerate(phases.items()):
            axes[row, 0].imshow(true_phase.cpu().numpy(), cmap='twilight')
            axes[row, 0].set_title(titles[0])
            axes[row, 0].axis('off')
            axes[row, 1].imshow(phase.cpu().numpy(), cmap='twilight')
            axes[row, 1].set_title(f"{titles[1]} - {key}")
            axes[row, 1].axis('off')
            aligned = phase - phase.mean() + true_phase.mean()
            error = (aligned - true_phase).cpu().numpy()
            axes[row, 2].imshow(error, cmap='RdBu', vmin=-1, vmax=1)
            axes[row, 2].set_title(f"{titles[2]} - {key}")
            axes[row, 2].axis('off')
        plt.tight_layout()
        plt.show()
        print("\n" + "="*60)
        print("实验结果总结")
        print("="*60)
        metrics_dict = {
            '有边界条件': results['metrics_with_boundary'],
            '无边界条件（好初始猜测）': results['metrics_no_boundary_good'],
            '无边界条件（随机初始猜测）': results['metrics_no_boundary_random']
        }
        for key, metrics in metrics_dict.items():
            mse, corr = metrics[1], metrics[2]
            print(f"{key}: MSE = {mse:.6e}, 相关系数 = {corr:.4f}")
            print("-" * 40)
    
    def run_comparison_experiments(self):
        print(f"使用设备: {self.device}")
        true_phase, intensity = self.create_test_data()
        target_dI_dz = self.compute_target_dI_dz(true_phase, intensity)
        print(f"真实相位范围: [{true_phase.min():.3f}, {true_phase.max():.3f}]")
        results = {}
        print("\n" + "="*60)
        print("实验1: 有边界条件优化")
        print("="*60)
        phase_with_boundary = self.optimize_with_boundary(
            true_phase, intensity, target_dI_dz
        )
        results['phase_with_boundary'] = phase_with_boundary
        results['metrics_with_boundary'] = self.analyze_result(
            phase_with_boundary, true_phase, "有边界条件"
        )
        print("\n" + "="*60)
        print("实验2: 无边界条件优化（高斯模糊初始猜测）")
        print("="*60)
        phase_no_boundary_good = self.optimize_without_boundary(
            true_phase, intensity, target_dI_dz,
            init_method='gaussian_smooth'
        )
        results['phase_no_boundary_good'] = phase_no_boundary_good
        results['metrics_no_boundary_good'] = self.analyze_result(
            phase_no_boundary_good, true_phase, "无边界条件（好初始猜测）"
        )
        print("\n" + "="*60)
        print("实验3: 无边界条件优化（随机初始猜测）")
        print("="*60)
        phase_no_boundary_random = self.optimize_without_boundary(
            true_phase, intensity, target_dI_dz,
            init_method='random'
        )
        results['phase_no_boundary_random'] = phase_no_boundary_random
        results['metrics_no_boundary_random'] = self.analyze_result(
            phase_no_boundary_random, true_phase, "无边界条件（随机初始猜测）"
        )
        self.plot_results(true_phase, results)
        return true_phase, results

def main():
    solver = TIESolver(image_size=128)
    true_phase, results = solver.run_comparison_experiments()
    return true_phase, results

if __name__ == "__main__":
    true_phase, results = main()