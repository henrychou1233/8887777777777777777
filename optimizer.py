import torch
from torch.optim import Adam, AdamW, Adamax
import math
from typing import List, Optional

def adamax(params: List[torch.Tensor],
           grads: List[torch.Tensor],
           exp_avgs: List[torch.Tensor],
           exp_infs: List[torch.Tensor],
           state_steps: List[int],
           *,
           beta1: float,
           beta2: float,
           lr: float,
           weight_decay: float,
           eps: float,
           maximize: bool):
    """
    功能性 API，執行 AdaMax 算法的計算。
    """
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_inf = exp_infs[i]
        step = state_steps[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # 更新偏差校正後的第一動量估計
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # 更新指數加權的無窮範數
        norm_buf = torch.cat([
            exp_inf.mul_(beta2).unsqueeze(0),
            grad.abs().add_(eps).unsqueeze_(0)
        ], 0)
        torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)

        bias_correction = 1 - beta1 ** step
        step_size = lr / bias_correction

        param.addcdiv_(exp_avg, exp_inf, value=-step_size)

class AdaMaxW(Adamax):
    """
    實現 AdaMax 算法，並加入解耦的權重衰減（weight decay）以及學習率 warm-up 和 cosine decay。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, maximize=False,
                 warmup_steps: int = 0, total_steps: int = 10000,
                 max_lr: Optional[float] = None, grad_clip: Optional[float] = None):
        if not 0.0 <= lr:
            raise ValueError(f"無效的學習率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"無效的 epsilon 值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"無效的 beta 參數 (索引 0): {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"無效的 beta 參數 (索引 1): {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"無效的 weight_decay 值: {weight_decay}")

        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, maximize=maximize)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.global_step = 0
        self.max_lr = max_lr if max_lr is not None else lr  # 學習率上限
        self.grad_clip = grad_clip  # 梯度裁剪

        # 保存初始學習率到每個參數組中
        for group in self.param_groups:
            group.setdefault('base_lr', lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 更新全局步數，並根據步數調整學習率
        self.global_step += 1
        for group in self.param_groups:
            base_lr = group.get('base_lr', group['lr'])
            if self.global_step < self.warmup_steps:
                # 線性 warm-up：從 0 緩慢增加到 base_lr
                lr = base_lr * self.global_step / self.warmup_steps
            else:
                # cosine decay 衰減：讓學習率在整個訓練周期內平滑下降
                progress = (self.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            group['lr'] = min(lr, self.max_lr)  # 確保學習率不超過上限

            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_infs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.grad_clip is not None:  # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(p, self.grad_clip)
                params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError('AdaMaxW 不支持稀疏梯度')
                grads.append(grad)

                state = self.state[p]

                # 延遲初始化狀態變量
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_infs.append(state['exp_inf'])
                state['step'] += 1
                state_steps.append(state['step'])

            adamax(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_infs,
                   state_steps,
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'],
                   maximize=group['maximize'])

        return loss

def build_optimizer(model, config):
    """
    構建優化器，支持 Adam、AdamW 和 AdaMaxW。
    """
    lr = config.model.learning_rate
    weight_decay = config.model.weight_decay
    optimizer_name = config.model.optimizer

    if optimizer_name == "Adam":
        optimizer = Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_name in ["AdamW", "AdaMaxW"]:
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        if optimizer_name == "AdamW":
            optimizer = AdamW(
                parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
                amsgrad=True
            )
        else:  # AdaMaxW
            # 可在配置中額外設定 warmup_steps 與 total_steps，若未設定則使用預設值
            warmup_steps = getattr(config.model, 'warmup_steps', 0)
            total_steps = getattr(config.model, 'total_steps', 10000)
            optimizer = AdaMaxW(
                parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                total_steps=total_steps
            )
    else:
        raise ValueError(f"不支持的優化器: {optimizer_name}")

    return optimizer