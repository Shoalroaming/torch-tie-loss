import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftfreq

class TIELoss(nn.Module):
    """
    基于光强传输方程(TIE)的损失函数，支持mae，mse，rmse
    支持自动维度匹配，输入可以是 (N,N), (B,N,N) 或 (B,1,N,N)
    """
    def __init__(
        self,
        image_size: int,               # 只支持正方形输入
        wavelength_m: float = 632.8e-9,
        pixel_size_m: float = 8e-6,
        loss_type: str = 'mse',
    ):
        super().__init__()
        self.N = image_size
        self.wavelength_m = wavelength_m
        self.pixel_size_m = pixel_size_m
        self.loss_type = loss_type
        
        self.k = 2 * torch.pi / self.wavelength_m
        
        self.register_buffer('KX', None)
        self.register_buffer('KY', None)
        self.register_buffer('k_sq', None)
        self._create_freq_grids()
    
    def _create_freq_grids(self):
        fx = fftfreq(self.N, d=self.pixel_size_m, device=self.KX.device if self.KX is not None else None)
        fy = fftfreq(self.N, d=self.pixel_size_m, device=self.KX.device if self.KX is not None else None)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        KX = 2.0 * torch.pi * FX
        KY = 2.0 * torch.pi * FY
        k_sq = KX**2 + KY**2
        k_sq[0, 0] = 1.0
        
        self.register_buffer('KX', KX)
        self.register_buffer('KY', KY)
        self.register_buffer('k_sq', k_sq)
    
    def _gradient(self, f: torch.Tensor) -> tuple:
        F_f = fft2(f, norm='ortho')
        fdx = ifft2(1j * self.KX * F_f, norm='ortho').real
        fdy = ifft2(1j * self.KY * F_f, norm='ortho').real
        return fdx, fdy
    
    def _divergence(self, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        F_fx = fft2(fx, norm='ortho')
        F_fy = fft2(fy, norm='ortho')
        div_f = ifft2(1j * self.KX * F_fx + 1j * self.KY * F_fy, norm='ortho').real
        return div_f
    
    def _validate_and_reshape(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        shape = tensor.shape
        ndim = len(shape)
        
        if ndim == 2:  # (N, N) -> (1, 1, N, N)
            return tensor.unsqueeze(0).unsqueeze(0)
        elif ndim == 3:  # (B, N, N) -> (B, 1, N, N)
            return tensor.unsqueeze(1)
        elif ndim == 4:  # (B, 1, N, N)
            if shape[1] != 1:
                raise ValueError(f"{name} 的通道数必须为1，但得到 shape {shape}")
            return tensor
        else:
            raise ValueError(f"{name} 维度不合法，期望2、3或4维，但得到 shape {shape}")
    
    def forward(
        self,
        pred_phase: torch.Tensor,
        intensity: torch.Tensor,
        target_dI_dz: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播，基于光强传输方程(TIE)计算损失
        
        参数:
            pred_phase: 预测的相位分布, 支持 shape (N,N), (B,N,N) 或 (B,1,N,N)
            intensity: 光强分布, 支持 shape (N,N), (B,N,N) 或 (B,1,N,N)
            target_dI_dz: 实际的光强轴向导数分布, 支持 shape (N,N), (B,N,N) 或 (B,1,N,N)
        
        返回:
            标量损失值
        """
        pred_phase = self._validate_and_reshape(pred_phase, "pred_phase")
        intensity = self._validate_and_reshape(intensity, "intensity")
        target_dI_dz = self._validate_and_reshape(target_dI_dz, "target_dI_dz")
        
        phase_grad_x, phase_grad_y = self._gradient(pred_phase.squeeze(1))
        phase_grad_x = phase_grad_x.unsqueeze(1)
        phase_grad_y = phase_grad_y.unsqueeze(1)
        I_grad_phi_x = intensity * phase_grad_x
        I_grad_phi_y = intensity * phase_grad_y
        div_I_grad_phi = self._divergence(I_grad_phi_x.squeeze(1), I_grad_phi_y.squeeze(1))
        dI_dz = div_I_grad_phi.unsqueeze(1)/(-self.k) 

        diff = dI_dz - target_dI_dz
        
        if self.loss_type == 'mse':
            return F.mse_loss(diff, torch.zeros_like(diff))
        elif self.loss_type == 'mae':
            return F.l1_loss(diff, torch.zeros_like(diff))
        elif self.loss_type == 'rmse':
            return torch.sqrt(F.mse_loss(diff, torch.zeros_like(diff)))
        else:
            raise ValueError(f"不支持损失类型: {self.loss_type}，请选择 'mse', 'mae' 或 'rmse'")