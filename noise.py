import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class AdaptiveGaussianConfig:
    """自適應高斯噪聲產生器設定"""
    base_std: float = 1.0               # 基準標準差
    min_std: float = 0.01               # 最小標準差
    max_std: float = 2.0                # 最大標準差
    adaptation_rate: float = 0.01       # 初始自適應速率
    history_size: int = 100             # 歷史記錄大小
    
    # 加強自適應參數
    signal_momentum: float = 0.95       # 訊號 EMA 動量
    noise_momentum: float = 0.9         # 噪聲 EMA 動量
    dynamic_scaling: bool = True        # 是否啟用動態縮放
    frequency_based: bool = True        # 是否啟用頻譜分析
    
    # 訊號頻譜分析參數
    freq_bands: int = 8                 # 頻段數量
    band_weights: Optional[List[float]] = None  # 頻段權重列表（若為 None 則均勻分配）
    dynamic_band_weights: bool = True   # 是否動態更新頻段權重
    
    # 穩定性與異常值控制
    stability_threshold: float = 0.1    # 穩定性門檻
    outlier_threshold: float = 2.5      # 異常值門檻（倍數）
    min_adaptation_rate: float = 0.001  # 最小自適應速率
    max_adaptation_rate: float = 0.1    # 最大自適應速率

class AdaptiveGaussianNoise:
    def __init__(self, config, noise_config: Optional[AdaptiveGaussianConfig] = None):
        """
        初始化自適應高斯噪聲產生器
        
        Args:
            config: 包含模型 device 等主要設定的配置物件
            noise_config: 噪聲產生器的設定參數
        """
        self.config = config
        self.device = config.model.device  # 從主配置中取得 device
        self.noise_config = noise_config or AdaptiveGaussianConfig()
        
        # 初始化訊號與噪聲統計追蹤
        self.initialize_trackers()
        
        # 如果啟用頻譜分析，則初始化相關元件
        if self.noise_config.frequency_based:
            self.initialize_frequency_analysis()

    def initialize_trackers(self) -> None:
        """初始化統計追蹤器與 EMA 變數"""
        self.signal_history = torch.zeros(
            (self.noise_config.history_size,),
            device=self.device
        )
        self.noise_history = torch.zeros(
            (self.noise_config.history_size,),
            device=self.device
        )
        
        # 訊號與噪聲標準差的 EMA 參數
        self.ema_signal_std: Optional[torch.Tensor] = None
        self.ema_noise_std: Optional[torch.Tensor] = None
        
        # 歷史記錄管理
        self.history_index = 0
        self.history_full = False
        self.current_adaptation_rate = self.noise_config.adaptation_rate

    def initialize_frequency_analysis(self) -> None:
        """初始化頻譜分析所需的頻段權重"""
        if self.noise_config.band_weights is None:
            # 若未指定，使用均勻權重
            weights = torch.ones(self.noise_config.freq_bands, device=self.device)
            self.band_weights = weights / weights.sum()
        else:
            self.band_weights = torch.tensor(
                self.noise_config.band_weights,
                device=self.device,
                dtype=torch.float32
            )
            self.band_weights = self.band_weights / self.band_weights.sum()

    def update_band_weights(self, signal_spectrum: torch.Tensor) -> None:
        """根據訊號特性更新頻段權重"""
        if not self.noise_config.dynamic_band_weights:
            return
        
        # 將頻譜沿最後一維切成多個頻段
        bands = torch.chunk(signal_spectrum, self.noise_config.freq_bands, dim=-1)
        # 計算每個頻段的平均能量
        band_energies = torch.stack([b.mean() for b in bands])
        # 正規化能量
        band_energies = band_energies / (band_energies.sum() + 1e-8)
        # 使用 EMA 更新頻段權重（動量 0.9 與 0.1）
        self.band_weights = 0.9 * self.band_weights + 0.1 * band_energies
        self.band_weights = self.band_weights / (self.band_weights.sum() + 1e-8)

    def compute_spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """計算訊號的頻譜特徵，用於頻譜因子計算"""
        # 計算 2D FFT 並取其幅值
        x_freq = torch.fft.fft2(x.float())
        spectrum = torch.abs(x_freq)
        
        # 頻譜正規化，避免最大值為零的情形
        max_val = torch.max(spectrum)
        normalized_spectrum = spectrum if max_val.item() == 0 else spectrum / (max_val + 1e-8)
        
        # 若啟用動態更新頻段權重則更新
        if self.noise_config.dynamic_band_weights:
            self.update_band_weights(normalized_spectrum)
        
        # 將正規化後的頻譜切成多個頻段並計算平均能量
        bands = torch.chunk(normalized_spectrum, self.noise_config.freq_bands, dim=-1)
        spectral_density = torch.stack([b.mean() for b in bands])
        
        return spectral_density

    def update_adaptation_rate(self, signal_stability: torch.Tensor) -> None:
        """根據訊號穩定性更新自適應速率"""
        if not self.noise_config.dynamic_scaling:
            return
        
        # 計算穩定因子（訊號穩定性與門檻比例，限制在一定範圍內）
        stability_factor = torch.clamp(
            signal_stability / self.noise_config.stability_threshold,
            0.1,
            10.0
        )
        
        # 自適應速率與穩定性成反比，並限制在設定範圍內
        self.current_adaptation_rate = torch.clamp(
            self.noise_config.adaptation_rate / stability_factor,
            self.noise_config.min_adaptation_rate,
            self.noise_config.max_adaptation_rate
        ).item()  # 轉換為 Python float

    def compute_signal_stability(self, x: torch.Tensor) -> torch.Tensor:
        """根據歷史訊號標準差的變化計算訊號穩定性"""
        if not self.history_full and self.history_index < 2:
            return torch.tensor(1.0, device=self.device)
        
        recent_signals = self.signal_history if self.history_full else self.signal_history[:self.history_index]
        if recent_signals.numel() < 2:
            return torch.tensor(1.0, device=self.device)
        
        diffs = torch.diff(recent_signals)
        # 使用差分標準差計算穩定性，數值越大代表變化越劇烈
        stability = 1.0 / (1.0 + torch.std(diffs) + 1e-8)
        return stability

    def compute_noise_stability(self) -> torch.Tensor:
        """根據歷史噪聲標準差的變化計算噪聲穩定性"""
        if not self.history_full and self.history_index < 2:
            return torch.tensor(1.0, device=self.device)
        
        recent_noises = self.noise_history if self.history_full else self.noise_history[:self.history_index]
        if recent_noises.numel() < 2:
            return torch.tensor(1.0, device=self.device)
        
        diffs = torch.diff(recent_noises)
        stability = 1.0 / (1.0 + torch.std(diffs) + 1e-8)
        return stability

    def adjust_for_outliers(self, current_noise_std: torch.Tensor) -> float:
        """
        檢查當前噪聲標準差是否為異常值，若超出歷史中位數的倍數則降低自適應速率
        """
        if not self.history_full and self.history_index < 2:
            return self.current_adaptation_rate
        
        recent_noises = self.noise_history if self.history_full else self.noise_history[:self.history_index]
        median_noise = torch.median(recent_noises)
        # 若當前噪聲超出設定倍數，則視為異常，降低自適應速率 50%
        if current_noise_std > self.noise_config.outlier_threshold * median_noise:
            return self.current_adaptation_rate * 0.5
        return self.current_adaptation_rate

    def calculate_adaptive_std(self, x: torch.Tensor) -> torch.Tensor:
        """
        根據頻譜特徵與訊號／噪聲穩定性計算自適應標準差
        """
        # 計算當前訊號的標準差（展平後）
        current_signal_std = torch.std(x.flatten())
        
        # 更新訊號歷史紀錄
        self.signal_history[self.history_index] = current_signal_std
        
        # 計算訊號穩定性
        signal_stability = self.compute_signal_stability(x)
        # 根據訊號穩定性更新自適應速率
        self.update_adaptation_rate(signal_stability)
        
        # 頻譜因子：若啟用頻譜分析則計算頻譜密度加權和
        if self.noise_config.frequency_based:
            spectral_density = self.compute_spectral_features(x)
            spectral_factor = torch.sum(spectral_density * self.band_weights)
        else:
            spectral_factor = torch.tensor(1.0, device=self.device)
        
        # 更新 EMA 的訊號標準差
        if self.ema_signal_std is None:
            self.ema_signal_std = current_signal_std
        else:
            self.ema_signal_std = (
                self.noise_config.signal_momentum * self.ema_signal_std +
                (1 - self.noise_config.signal_momentum) * current_signal_std
            )
        
        # 新增：計算噪聲穩定性，並與訊號穩定性取平均作為最終穩定因子
        noise_stability = self.compute_noise_stability()
        combined_stability = (signal_stability + noise_stability) / 2.0
        
        # 初步計算自適應標準差，結合 EMA 訊號標準差、自適應速率、頻譜因子與穩定性因子
        adaptive_std = self.ema_signal_std * self.current_adaptation_rate * spectral_factor * combined_stability
        
        # 限制標準差在設定範圍內
        adaptive_std = torch.clamp(
            adaptive_std,
            self.noise_config.min_std,
            self.noise_config.max_std
        )
        
        return adaptive_std

    def generate(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        生成自適應高斯噪聲
        
        Args:
            x: 輸入張量
            seed: 可選隨機種子
            
        Returns:
            生成的噪聲張量
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 計算自適應標準差
        std = self.calculate_adaptive_std(x)
        
        # 生成噪聲
        noise = torch.randn_like(x, device=self.device) * std
        
        # 更新 EMA 的噪聲標準差
        current_noise_std = torch.std(noise.flatten())
        if self.ema_noise_std is None:
            self.ema_noise_std = current_noise_std
        else:
            self.ema_noise_std = (
                self.noise_config.noise_momentum * self.ema_noise_std +
                (1 - self.noise_config.noise_momentum) * current_noise_std
            )
        
        # 檢查是否有異常值，若有則調整自適應速率
        adjusted_rate = self.adjust_for_outliers(current_noise_std)
        if adjusted_rate < self.current_adaptation_rate:
            self.current_adaptation_rate = adjusted_rate
        
        # 更新噪聲歷史紀錄
        self.noise_history[self.history_index] = current_noise_std
        
        # 更新歷史索引與是否已滿
        self.history_index = (self.history_index + 1) % self.noise_config.history_size
        if self.history_index == 0:
            self.history_full = True
        
        return noise

def get_adaptive_gaussian_noise(
    x: torch.Tensor,
    config,
    noise_config: Optional[AdaptiveGaussianConfig] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    方便函數：產生自適應高斯噪聲
    
    Args:
        x: 輸入張量
        config: 包含模型 device 等資訊的配置物件
        noise_config: 可選的噪聲配置
        seed: 可選隨機種子
        
    Returns:
        生成的噪聲張量
    """
    generator = AdaptiveGaussianNoise(config, noise_config)
    return generator.generate(x, seed)