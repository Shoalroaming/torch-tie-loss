import torch
from tieloss import TIELoss

def mem():
    return torch.cuda.memory_allocated() / 1024**2

def peak():
    return torch.cuda.max_memory_allocated() / 1024**2

def test_single_size():
    print("\n--- 单尺寸测试 (256x256) ---")
    torch.cuda.empty_cache()
    
    loss_fn = TIELoss(
        image_size=256,
        wavelength_m=632.8e-9,
        pixel_size_m=8e-6,
        loss_type='mse'
    ).cuda()
    
    batch_size = 4
    image_size = 256
    
    pred_phase = torch.randn(batch_size, image_size, image_size, 
                           device='cuda', requires_grad=True)
    intensity = torch.rand(batch_size, image_size, image_size, 
                         device='cuda') * 0.5 + 0.5  # 光强应为正值
    target_dI_dz = torch.randn(batch_size, image_size, image_size, 
                             device='cuda')
    
    init_mem = mem()
    
    loss = loss_fn(pred_phase, intensity, target_dI_dz)
    loss.backward()
    
    print(f"批大小: {batch_size}, 图像尺寸: {image_size}x{image_size}")
    print(f"初始内存: {init_mem:.1f} MB")
    print(f"峰值内存: {peak():.1f} MB")
    print(f"最终内存: {mem():.1f} MB")
    
    del pred_phase, intensity, target_dI_dz, loss
    torch.cuda.empty_cache()

def test_multi_sizes():
    print("\n--- 多尺寸测试 ---")
    torch.cuda.empty_cache()
    
    configs = [
        {'size': 128, 'batch': 8},
        {'size': 256, 'batch': 4},
        {'size': 512, 'batch': 2},
        {'size': 1024, 'batch': 1},
    ]
    
    loss_functions = {}
    for config in configs:
        s = config['size']
        loss_functions[s] = TIELoss(
            image_size=s,
            wavelength_m=632.8e-9,
            pixel_size_m=8e-6,
            loss_type='mse'
        ).cuda()
    
    init_mem = mem()
    print(f"初始内存: {init_mem:.1f} MB")
    
    for config in configs:
        s = config['size']
        b = config['batch']
        
        print(f"\n测试尺寸: {s}x{s}, 批大小: {b}")
        
        pred_phase = torch.randn(b, s, s, device='cuda', requires_grad=True)
        intensity = torch.rand(b, s, s, device='cuda') * 0.5 + 0.5
        target_dI_dz = torch.randn(b, s, s, device='cuda')
        
        loss = loss_functions[s](pred_phase, intensity, target_dI_dz)
        loss.backward()
        
        print(f"  当前内存: {mem():.1f} MB, 峰值内存: {peak():.1f} MB, 最终内存: {mem():.1f} MB")
        
        del pred_phase, intensity, target_dI_dz, loss
        torch.cuda.empty_cache()
    

    del loss_functions
    torch.cuda.empty_cache()
    
def test_various_batch_sizes():
    print("\n--- 不同批大小测试 ---")
    torch.cuda.empty_cache()
    
    image_size = 256
    batch_sizes = [1, 2, 4, 8, 16]
    
    loss_fn = TIELoss(
        image_size=image_size,
        wavelength_m=632.8e-9,
        pixel_size_m=8e-6,
        loss_type='mse'
    ).cuda()
    
    print(f"图像尺寸: {image_size}x{image_size}")
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        init_mem = mem()
        print(f"\n批大小: {batch_size}")
        print(f"初始内存: {init_mem:.1f} MB")
        
        pred_phase = torch.randn(batch_size, image_size, image_size, 
                               device='cuda', requires_grad=True)
        intensity = torch.rand(batch_size, image_size, image_size, 
                             device='cuda') * 0.5 + 0.5
        target_dI_dz = torch.randn(batch_size, image_size, image_size, 
                                  device='cuda')
        
        loss = loss_fn(pred_phase, intensity, target_dI_dz)
        loss.backward()
        
        print(f"  当前内存: {mem():.1f} MB, 峰值内存: {peak():.1f} MB, 最终内存: {mem():.1f} MB")
        
        del pred_phase, intensity, target_dI_dz, loss
        torch.cuda.empty_cache()

def main():
    test_single_size()
    test_multi_sizes()
    test_various_batch_sizes()

if __name__ == "__main__":
    main()