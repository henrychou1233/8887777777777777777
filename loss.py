import torch
import torch.nn.functional as F

def get_loss(model, constant_dict, x_0, t, config):
    """
    改進的損失函數，確保數值範圍與原版相似：
    - 保持原有MSE Loss主體
    - 謹慎加入輔助損失項
    - 消除可能導致數值大幅增加的因素
    """
    device = getattr(config.model, 'device', 'cuda')
    x_0 = x_0.to(device, non_blocking=True)
    t = t.to(device)
    
    # 抓 alpha
    if 'alphas_cumprod' in constant_dict:
        at = constant_dict['alphas_cumprod'].to(device).index_select(0, t).view(-1, 1, 1, 1)
    else:
        b = constant_dict['betas'].to(device)
        at = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    
    # 模擬加噪影像
    e = torch.randn_like(x_0, device=device)
    x = at.sqrt() * x_0 + (1 - at).sqrt() * e
    
    # 混合精度控制（預設 False）
    use_amp = getattr(config.model, 'mixed_precision', False)
    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
        output = model(x, t.float())
        
        # SNR weighting（與原版保持一致）
        if getattr(config.model, 'snr_weighting', False):
            snr = at / (1 - at + 1e-8)
            weight = snr / (1 + snr)
            weight = torch.clamp(weight, min=0.01, max=1.0)
            mse_loss = F.mse_loss(output, e, reduction='none')
            mse_loss = (mse_loss * weight.view(-1, 1, 1, 1)).mean()
        else:
            mse_loss = F.mse_loss(output, e)
        
        # 極小的cosine loss輔助項（幾乎不影響總損失值）
        cosine_weight = getattr(config.model, 'cosine_loss_weight', 0.0)
        if cosine_weight > 0:
            # 計算cosine similarity
            norm_output = F.normalize(output, p=2, dim=1)
            norm_noise = F.normalize(e, p=2, dim=1)
            cosine_sim = (norm_output * norm_noise).sum(dim=1)
            # 使用非常小的權重
            actual_weight = min(cosine_weight, 0.01)  # 限制最大權重
            cosine_loss = (1 - cosine_sim.mean()) * actual_weight
        else:
            cosine_loss = torch.tensor(0.0, device=device)
        
        # 小幅平滑化處理
        smoothness = getattr(config.model, 'smoothness', 0.0)
        if smoothness > 0:
            # 計算相鄰像素間差異
            diff_x = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:]).mean()
            diff_y = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]).mean()
            smooth_loss = (diff_x + diff_y) * min(smoothness, 0.01)  # 極小權重
        else:
            smooth_loss = torch.tensor(0.0, device=device)
        
        # 確保主要損失項仍是MSE
        total_loss = mse_loss + cosine_loss + smooth_loss
    
    # debug 輸出
    if getattr(config.model, 'debug', False):
        print(f"[Loss] Total: {total_loss.item():.6f} | MSE: {mse_loss.item():.6f} | "
              f"Cosine: {cosine_loss.item():.6f} | Smooth: {smooth_loss.item():.6f}")
    
    return total_loss