import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    計算 Sigmoid 函數，並考慮數值穩定性
    Args:
        x: 輸入陣列
    Returns:
        輸入陣列經 Sigmoid 映射後的值
    """
    # 利用 np.where 依據 x 的正負分支計算，避免數值下溢
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

@dataclass
class AdaptiveParams:
    """自適應調度參數設定"""
    min_adaptation_factor: float = 0.5   # 最小適應因子
    max_adaptation_factor: float = 2.0   # 最大適應因子
    min_temperature: float = 0.3         # 最低溫度
    max_temperature: float = 1.5         # 最高溫度
    early_phase_steps: int = 1000        # 初期階段步數
    late_phase_steps: int = 1000         # 後期階段步數
    adjustment_interval: int = 100       # 調整間隔步數
    momentum: float = 0.9                # 動量係數
    learning_rate: float = 0.01          # 學習率

class AdaptiveSigmoidScheduler:
    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        adaptation_factor: float = 1.2,
        temperature: float = 0.8,
        optimize_params: bool = False,
        adaptive_params: Optional[AdaptiveParams] = None
    ):
        """
        初始化自適應 Sigmoid Beta 調度器
        
        參數:
            num_timesteps: 擴散步數
            beta_start: 起始 beta 值
            beta_end: 結束 beta 值
            adaptation_factor: 控制曲線陡峭度的因子
            temperature: 控制曲線平滑度的參數
            optimize_params: 是否進行參數優化（網格搜尋）
            adaptive_params: 自適應調度參數設定
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.adaptation_factor = adaptation_factor
        self.temperature = temperature
        self.adaptive_params = adaptive_params or AdaptiveParams()
        
        # 初始化追蹤變數
        self.step_counter = 0
        self.parameter_history = []
        self.phase_metrics = {'early': [], 'middle': [], 'late': []}
        self.velocity = {'af': 0.0, 'temp': 0.0}
        
        # 若需要參數優化，則使用網格搜尋決定最佳參數
        if optimize_params:
            self.adaptation_factor, self.temperature = self._optimize_parameters()
            
        self.betas = self._compute_betas()
        
    def _compute_phase_adjustments(self) -> Tuple[float, float]:
        """
        根據目前步數計算階段性調整係數
        Returns:
            (適應因子乘數, 溫度乘數)
        """
        if self.step_counter < self.adaptive_params.early_phase_steps:
            # 初期階段：較積極的調整
            return 1.2, 0.8
        elif self.step_counter > (self.num_timesteps - self.adaptive_params.late_phase_steps):
            # 後期階段：較保守的調整
            return 0.8, 1.2
        else:
            # 中期階段：標準參數
            return 1.0, 1.0

    def _apply_momentum_update(self, current_value: float, gradient: float, velocity_key: str) -> float:
        """
        利用動量更新參數
        Args:
            current_value: 當前參數值
            gradient: 計算得到的梯度值
            velocity_key: 用於追蹤該參數的動量字典鍵
        Returns:
            更新後的參數值
        """
        self.velocity[velocity_key] = (
            self.adaptive_params.momentum * self.velocity[velocity_key] +
            self.adaptive_params.learning_rate * gradient
        )
        updated_value = current_value - self.velocity[velocity_key]
        return updated_value

    def _compute_betas(self) -> torch.Tensor:
        """
        利用自適應 Sigmoid 曲線計算 beta 值序列
        Returns:
            beta 值張量
        """
        # 生成在 [-6, 6] 範圍內均勻分布的數值，作為 Sigmoid 的輸入
        x = np.linspace(-6, 6, self.num_timesteps)
        
        # 根據目前步數計算階段性乘數
        af_mult, temp_mult = self._compute_phase_adjustments()
        current_af = self.adaptation_factor * af_mult
        current_temp = self.temperature * temp_mult
        
        # 限制參數在設定範圍內
        current_af = np.clip(current_af, 
                             self.adaptive_params.min_adaptation_factor,
                             self.adaptive_params.max_adaptation_factor)
        current_temp = np.clip(current_temp,
                               self.adaptive_params.min_temperature,
                               self.adaptive_params.max_temperature)
        
        # 使用自適應溫度控制 Sigmoid 曲線，並依據適應因子調整曲線陡峭度
        base_sigmoid = sigmoid(x / current_temp)
        adapted_sigmoid = np.power(base_sigmoid, current_af)
        
        # 平滑處理：利用移動平均濾波器平滑數值曲線
        adapted_sigmoid = self._apply_smoothing(adapted_sigmoid)
        
        # 將平滑後的值縮放到 [beta_start, beta_end] 區間
        betas = adapted_sigmoid * (self.beta_end - self.beta_start) + self.beta_start
        # 為保證單調性，使用排序（理論上 Sigmoid 已單調，但此處作保險處理）
        return torch.tensor(np.sort(betas)).float()
    
    def _apply_smoothing(self, values: np.ndarray) -> np.ndarray:
        """
        對輸入曲線進行平滑處理，使用簡單移動平均濾波器
        Args:
            values: 輸入數值陣列
        Returns:
            平滑後的陣列
        """
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(values, kernel, mode='same')
        # 保留首尾數值
        smoothed[0] = values[0]
        smoothed[-1] = values[-1]
        return smoothed
    
    def _optimize_parameters(self) -> Tuple[float, float]:
        """
        透過網格搜尋優化適應因子與溫度參數
        Returns:
            (最佳適應因子, 最佳溫度)
        """
        best_score = float('inf')
        best_params = (self.adaptation_factor, self.temperature)
        
        # 為避免改變原有參數，採用局部變數進行網格搜尋
        for af in np.linspace(self.adaptive_params.min_adaptation_factor,
                              self.adaptive_params.max_adaptation_factor,
                              20):
            for temp in np.linspace(self.adaptive_params.min_temperature,
                                    self.adaptive_params.max_temperature,
                                    20):
                # 使用局部變數暫存
                test_af = af
                test_temp = temp
                # 根據當前測試參數計算 beta 值
                x = np.linspace(-6, 6, self.num_timesteps)
                base_sigmoid = sigmoid(x / test_temp)
                adapted_sigmoid = np.power(base_sigmoid, test_af)
                adapted_sigmoid = self._apply_smoothing(adapted_sigmoid)
                betas = adapted_sigmoid * (self.beta_end - self.beta_start) + self.beta_start
                betas = np.sort(betas)
                betas_tensor = torch.tensor(betas).float()
                
                # 評估指標：平滑度（第二階導數絕對值平均）、連續性（相鄰差值平均）、
                # 與起始及結束值誤差
                smoothness = torch.mean(torch.abs(torch.diff(betas_tensor, n=2)))
                coverage = torch.mean(torch.abs(torch.diff(betas_tensor)))
                start_end_adherence = abs(betas_tensor[0] - self.beta_start) + abs(betas_tensor[-1] - self.beta_end)
                
                # 組合評分（可根據實驗調整各項權重）
                score = 0.4 * smoothness + 0.4 * coverage + 0.2 * start_end_adherence
                
                if score < best_score:
                    best_score = score
                    best_params = (test_af, test_temp)
        
        return best_params
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        執行一步調整，若提供監控指標則依據動量更新參數
        Args:
            metrics: 可選的指標字典，包含 'smoothness', 'coverage', 'stability'
        """
        self.step_counter += 1
        
        if metrics and self.step_counter % self.adaptive_params.adjustment_interval == 0:
            # 根據提供的指標計算梯度（權重可依實際需求調整）
            af_gradient = (
                metrics.get('smoothness', 0) * 0.4 +
                metrics.get('coverage', 0) * 0.4 +
                metrics.get('stability', 0) * 0.2
            )
            temp_gradient = (
                metrics.get('smoothness', 0) * 0.3 +
                metrics.get('coverage', 0) * 0.3 +
                metrics.get('stability', 0) * 0.4
            )
            
            # 利用動量更新參數
            self.adaptation_factor = self._apply_momentum_update(self.adaptation_factor, af_gradient, 'af')
            self.temperature = self._apply_momentum_update(self.temperature, temp_gradient, 'temp')
            
            # 更新 beta 調度
            self.betas = self._compute_betas()
            
            # 紀錄參數變化歷史
            self.parameter_history.append({
                'step': self.step_counter,
                'adaptation_factor': self.adaptation_factor,
                'temperature': self.temperature,
                'metrics': metrics
            })
            
            # 根據步數將指標分配到不同階段
            if self.step_counter < self.adaptive_params.early_phase_steps:
                self.phase_metrics['early'].append(metrics)
            elif self.step_counter > (self.num_timesteps - self.adaptive_params.late_phase_steps):
                self.phase_metrics['late'].append(metrics)
            else:
                self.phase_metrics['middle'].append(metrics)
    
    def get_betas(self) -> torch.Tensor:
        """取得目前的 beta 調度張量"""
        return self.betas
    
    def get_alphas(self) -> torch.Tensor:
        """取得 alpha 調度張量（1 - beta）"""
        return 1 - self.betas
    
    def get_alphas_cumprod(self) -> torch.Tensor:
        """取得 alpha 累積乘積張量"""
        return torch.cumprod(self.get_alphas(), dim=0)
    
    def get_current_parameters(self) -> Dict[str, float]:
        """取得目前自適應參數值與步數"""
        return {
            'adaptation_factor': self.adaptation_factor,
            'temperature': self.temperature,
            'step': self.step_counter
        }
    
    def get_parameter_history(self) -> list:
        """取得參數更新歷史紀錄"""
        return self.parameter_history
    
    def get_phase_metrics(self) -> Dict[str, list]:
        """取得各階段的指標紀錄"""
        return self.phase_metrics

def beta_schedule(beta_schedule: str, beta_start: float, beta_end: float, num_diffusion_timesteps: int) -> torch.Tensor:
    """
    根據指定類型建立 beta 調度，用於擴散模型
    
    參數:
        beta_schedule: 調度類型 ("quad", "linear", "const", "jsd", "sigmoid", "adapt_sigmoid")
        beta_start: 起始 beta 值
        beta_end: 結束 beta 值
        num_diffusion_timesteps: 擴散步數
    Returns:
        beta 值張量
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            ) ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "adapt_sigmoid":
        scheduler = AdaptiveSigmoidScheduler(
            num_timesteps=num_diffusion_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            optimize_params=True
        )
        betas = scheduler.get_betas()
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")
    
    return torch.tensor(np.sort(betas)).float()

def compute_alpha(beta: torch.Tensor, t: torch.Tensor, config: object) -> torch.Tensor:
    """
    計算擴散過程中 alpha 值
    Args:
        beta: beta 調度張量
        t: 時間步張量
        config: 包含模型 device 的配置物件
    Returns:
        alpha 值張量
    """
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    beta = beta.to(config.model.device)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_alpha2(beta: torch.Tensor, t: torch.Tensor, config: object) -> torch.Tensor:
    """
    使用明確型別轉換計算 alpha 值
    Args:
        beta: beta 調度張量
        t: 時間步張量
        config: 包含模型 device 的配置物件
    Returns:
        alpha 值張量
    """
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    beta = beta.to(config.model.device)
    t = t.to(config.model.device).long()  # 確保 t 為 long 型別
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int], config: object) -> torch.Tensor:
    """
    根據批次維度從列表中取得特定索引值
    Args:
        vals: 值列表（張量）
        t: 時間步張量
        x_shape: 輸入張量形狀
        config: 配置物件
    Returns:
        取索引後的張量
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)