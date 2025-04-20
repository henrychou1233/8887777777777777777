import torch
import pickle
import numpy as np
import os
import time
from datetime import timedelta
from asyncio import constants
from unet import *
from utilities import *
from forward_process import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import metric
from feature_extractor import *
from diffusers import AutoencoderKL
from sklearn.neighbors import NearestNeighbors
from consistencydecoder import ConsistencyDecoder
import torch.nn as nn
import torchvision.transforms as transforms
from functools import partial
from sklearn.model_selection import KFold
from scipy.stats import iqr

def adaptive_metric(x, y, alpha, beta):
    """
    全局定義的 adaptive 距離函數，用來計算：
        distance = alpha * L1_distance + beta * L2_distance
    """
    l1_distance = np.sum(np.abs(x - y))
    l2_distance = np.sqrt(np.sum((x - y) ** 2))
    return alpha * l1_distance + beta * l2_distance

def get_adaptive_weights(X, k):
    """
    根據訓練集 X（numpy 陣列）計算 k 最近鄰下的全局平均 L1 與 L2 距離，
    並返回正規化後的權重 alpha 與 beta，使得 alpha + beta = 1。
    """
    # 使用純 L1 度量
    nn_l1 = NearestNeighbors(n_neighbors=k, metric="l1")
    nn_l1.fit(X)
    dists_l1, _ = nn_l1.kneighbors(X)
    avg_l1 = np.mean(dists_l1)
    
    # 使用純 L2 度量
    nn_l2 = NearestNeighbors(n_neighbors=k, metric="l2")
    nn_l2.fit(X)
    dists_l2, _ = nn_l2.kneighbors(X)
    avg_l2 = np.mean(dists_l2)
    
    eps = 1e-8  # 避免除零
    w1 = 1.0 / (avg_l1 + eps)
    w2 = 1.0 / (avg_l2 + eps)
    sum_w = w1 + w2
    alpha = w1 / sum_w
    beta = w2 / sum_w
    return alpha, beta

class KNN:
    def __init__(self, config, k=None, num_bins=10):
        self.config = config
        self.num_bins = num_bins
        
        # 使用配置中的默認值或初始化時提供的值
        if k is None:
            self.k = getattr(config.model, "knn_k", 5)
        else:
            self.k = k
            
        print(f"Using k value: k = {self.k}")
        
        # 根據 config 選擇距離度量類型
        if config.model.KNN_metric == "adaptive":
            self.metric_type = "adaptive"
            self.model = None  # 在 fit 時初始化
            self.alpha = None
            self.beta = None
        elif config.model.KNN_metric == "l1+l2":
            self.metric_type = "fixed"
            # 固定權重從 config 讀取，預設值為 0.5 與 0.5
            alpha = getattr(config.model, "alpha", 0.5)
            beta = getattr(config.model, "beta", 0.5)
            self.model = None  # 在 fit 時初始化
        else:
            self.metric_type = "other"
            self.model = None  # 在 fit 時初始化
    
    def fit(self, X):
        """
        X 為訓練資料特徵（Tensor 或 numpy 陣列）。
        
        如果使用自適應距離，先計算全局 L1 與 L2 平均值以獲得權重，
        然後初始化 NearestNeighbors 模型。
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X
        
        if self.metric_type == "adaptive":
            # 計算自適應權重
            self.alpha, self.beta = get_adaptive_weights(X_np, self.k)
            print(f"Adaptive weights computed: alpha = {self.alpha}, beta = {self.beta}")
            # 利用 partial 定義距離函數，避免使用 lambda
            metric_fn = partial(adaptive_metric, alpha=self.alpha, beta=self.beta)
            self.model = NearestNeighbors(n_neighbors=self.k, metric=metric_fn)
        elif self.metric_type == "fixed":
            # 固定權重
            alpha = getattr(self.config.model, "alpha", 0.5)
            beta = getattr(self.config.model, "beta", 0.5)
            metric_fn = partial(adaptive_metric, alpha=alpha, beta=beta)
            self.model = NearestNeighbors(n_neighbors=self.k, metric=metric_fn)
        else:
            # 其他標準距離度量
            self.model = NearestNeighbors(n_neighbors=self.k, metric=self.config.model.KNN_metric)
        
        self.model.fit(X_np)
        
        distances, _ = self.model.kneighbors(X_np)
        avg_distances = distances.mean(axis=1)
        self.histogram, self.bin_edges = np.histogram(avg_distances, self.num_bins)
        
        print(f"bin edges: {self.bin_edges}")
        print(f"histogram: {self.histogram}")
    
    def transform(self, X):
        """
        轉換方法，檢查 X 是否為 numpy array，若不是則轉換
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X
        
        distances, indices = self.model.kneighbors(X_np)
        return distances, indices

def get_bins_and_mappings(knn, distances, indices):
    """
    根據每個樣本的鄰近點距離平均值，利用預先定義的 bin_edges 做分箱，
    並返回對應的映射與鍵值。
    """
    mappings = []
    keys = []
    for i in range(distances.shape[0]):
        avg_distance = np.mean(distances[i])
        bin_id = np.digitize(avg_distance, knn.bin_edges, right=True) - 1
        bin_id = min(bin_id, len(knn.bin_edges) - 2) + 1
        keys.append(bin_id)
        mapping = {bin_id: [ind.item() for ind in indices[i]]}
        mappings.append(mapping)
    return mappings, keys

def validate(unet, constants_dict, config):

    if config.data.name in ['BTAD', 'VisA', 'MVTec']:
        test_dataset = MVTecDataset(
            root=config.data.data_dir,
            category=config.data.category,
            config=config,
            is_train=False,
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=False,
        )

        train_dataset = MVTecDataset(
            root=config.data.data_dir,
            category=config.data.category,
            config=config,
            is_train=True,
        )
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.DA_batch_size,
            shuffle=True,
            num_workers=config.model.num_workers,
            drop_last=False,
        )

    labels_list = []
    predictions = []
    anomaly_map_list = []
    GT_list = []
    reconstructed_list = []
    forward_list = []
    forward_list_orig = []
    l1_latent_list = []
    cos_dist_list = []
    step_list = []
    filename_list = []
    anomaly_map_recon_list = []
    anomaly_map_feature_list = []
    anomaly_map_latent_list = []

    if config.model.latent:
        if config.model.latent_backbone == "VAE":
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            vae.to(config.model.device)
            if config.model.consistency_decoder:
                consistency_decoder = ConsistencyDecoder(device=config.model.device)
            else:
                consistency_decoder = False
            vae.eval()
        else:
            print("Error: backbone needs to be VAE")

        if config.model.dynamic_steps or (config.model.distance_metric_eval == "combined"):
            if config.model.fe_backbone == "wide_resnet50":
                feature_extractor = wide_resnet50_2(pretrained=True)[0]
            elif config.model.fe_backbone == "resnet34":
                feature_extractor = resnet34(pretrained=True)[0]
            elif config.model.fe_backbone == "resnet101":
                feature_extractor = resnet101(pretrained=True)[0]
            elif config.model.fe_backbone == "wide_resnet101":
                feature_extractor = wide_resnet101_2(pretrained=True)[0]
            else:
                print("Error: no valid fe backbone selected")
            feature_extractor.to(config.model.device)
            feature_extractor = Domain_adaptation(unet, feature_extractor, vae, config,
                                                  fine_tune=config.model.DA_fine_tune,
                                                  constants_dict=constants_dict,
                                                  dataloader=trainloader,
                                                  consistency_decoder=consistency_decoder)
            feature_extractor.eval()

            knn_transform = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # 初始化 KNN 模型，使用固定 k 值
            knn = KNN(config=config, k=config.model.knn_k, num_bins=10)

            # 將訓練資料的特徵堆疊起來
            train_stack = []
            del vae
            torch.cuda.empty_cache()

            for i, train_batch in enumerate(trainloader):
                train_batch = knn_transform(train_batch[0])
                train_batch = feature_extractor(train_batch.to(config.model.device))
                selected_features = [train_batch[i] for i in config.model.selected_features]
                common_size = (16, 16)
                adaptive_pool = nn.AdaptiveAvgPool2d(common_size)
                pooled_features = [adaptive_pool(feature_map) for feature_map in selected_features]
                flattened_features = [pf.view(pf.size(0), -1) for pf in pooled_features]
                train_batch = torch.cat(flattened_features, dim=1)
                train_stack.append(train_batch.detach().cpu())
                torch.cuda.empty_cache()

            # 擬合 KNN 模型
            concatenated_features = torch.cat(train_stack, dim=0)
            knn.fit(concatenated_features)

            knnPickle = open(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category,
                                            f"knn_{knn.k}_{config.model.DA_epochs}"), 'wb')
            del train_stack
            del trainloader
            torch.cuda.empty_cache()
            pickle.dump(knn, knnPickle)
            knnPickle.close()

    def roundup(x, n=10):
        res = np.ceil(x / n) * n
        mask = np.logical_and(x % n < n / 2, x % n > 0)
        res[mask] -= n
        return res

    if config.model.latent_backbone == "VAE":
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.to(config.model.device)
        vae.eval()

    if config.data.name in ['BTAD', 'VisA', 'MVTec']:
        with torch.no_grad():
            start = time.time()
            for step, (data, targets, labels, filename) in enumerate(testloader):
                data_placeholder = data

                if config.model.dynamic_steps:
                    test_batch = data
                    test_batch = knn_transform(test_batch)
                    test_batch = feature_extractor(test_batch.to(config.model.device))
                    selected_features = [test_batch[i] for i in config.model.selected_features]
                    adaptive_pool = nn.AdaptiveAvgPool2d(common_size)
                    pooled_features = [adaptive_pool(feature_map) for feature_map in selected_features]
                    flattened_features = [pf.view(pf.size(0), -1) for pf in pooled_features]
                    test_batch = torch.cat(flattened_features, dim=1)
                    test_batch = test_batch.detach().cpu().numpy()
                    torch.cuda.empty_cache()

                    distances, indices = knn.transform(test_batch)
                    mappings, keys = get_bins_and_mappings(knn, distances, indices)
                    mapping_int = int(list(set(mappings[0].keys()))[0])
                    bin_ids_array = np.array(keys)
                    step_sizes_array = np.maximum(bin_ids_array, 2) / 10 * config.model.test_trajectoy_steps
                    step_size = roundup(step_sizes_array)
                    skip = np.maximum(step_size / 10, 1).astype(int)
                    step_list.extend(step_size)
                else:
                    step_size = config.model.test_trajectoy_steps
                    skip = config.model.skip

                filename_list.append(filename)
                forward_list_orig.append(data)
                forward_list.append(data)
                if config.model.latent:
                    data = data.to(config.model.device)
                    data = vae.encode(data).latent_dist.sample() * 0.18215

                test_trajectoy_steps = torch.Tensor([step_size]).type(torch.int64).to(config.model.device)[0]
                at = compute_alpha2(constants_dict['betas'], test_trajectoy_steps.long(), config)

                if config.model.noise_sampling:
                    noise = torch.randn_like(data).to(config.model.device)
                    noisy_image = at.sqrt() * data + (1 - at).sqrt() * noise
                else:
                    noisy_image = data
                    if config.model.downscale_first:
                        noisy_image = noisy_image * at.sqrt()

                if config.model.dynamic_steps:
                    seq = [torch.arange(0, end, step).to(test_trajectoy_steps.device) for end, step in zip(test_trajectoy_steps, skip)]
                else:
                    seq = range(0, test_trajectoy_steps, skip)

                if config.model.dynamic_steps:
                    reconstructed, rec_x0 = my_generalized_steps(data, noisy_image, seq, unet,
                                                                 constants_dict['betas'], config,
                                                                 eta2=config.model.eta2, eta3=0,
                                                                 constants_dict=constants_dict, eraly_stop=False)
                else:
                    reconstructed, rec_x0 = DA_generalized_steps(data, noisy_image, seq, unet,
                                                                 constants_dict['betas'], config,
                                                                 eta2=config.model.eta2, eta3=0,
                                                                 constants_dict=constants_dict, eraly_stop=False)

                data_reconstructed = reconstructed[-1].to(config.model.device)

                if config.model.latent_backbone == "VAE":
                    reconstructed = 1 / 0.18215 * data_reconstructed
                    if config.model.consistency_decoder:
                        reconstructed = consistency_decoder(reconstructed)
                    else:
                        reconstructed = vae.decode(reconstructed.to(config.model.device)).sample
                else:
                    print("Error: backbone needs to be VAE")

                l1_latent = color_distance(data_reconstructed, data, config, out_size=config.data.image_size)
                cos_dist = feature_distance_new(reconstructed, data_placeholder, feature_extractor, config)

                anomaly_map_latent = recon_heat_map(data_reconstructed, data, config)
                anomaly_map_feature = feature_heat_map(reconstructed, data_placeholder, feature_extractor, config)

                l1_latent_list.append(l1_latent)
                cos_dist_list.append(cos_dist)
                anomaly_map_latent_list.append(anomaly_map_latent)
                anomaly_map_feature_list.append(anomaly_map_feature)

                GT_list.append(targets)
                reconstructed_list.append(reconstructed)

                for label in labels:
                    labels_list.append(0 if label == 'good' else 1)

            end = time.time()
            print('Inference time is ', str(timedelta(seconds=end - start)))

            l1_latent_normalized_list = scale_values_between_zero_and_one(l1_latent_list)
            cos_dist_normalized_list = scale_values_between_zero_and_one(cos_dist_list)
            heatmap_latent_list = heatmap_latent(l1_latent_normalized_list, cos_dist_normalized_list, config)

            concat_heatmap = torch.cat(heatmap_latent_list, dim=0)
            predictions_normalized = [torch.max(heatmap).item() for heatmap in concat_heatmap]
            threshold = metric(labels_list, predictions_normalized, heatmap_latent_list, GT_list, config)
            print('threshold: ', threshold)

            reconstructed_list = torch.cat(reconstructed_list, dim=0)
            forward_list = torch.cat(forward_list, dim=0)
            if config.model.latent:
                forward_list_orig = torch.cat(forward_list_orig, dim=0)
                filename_list = [item for tup in filename_list for item in tup]
                anomaly_map_latent_list = torch.cat(anomaly_map_latent_list, dim=0)
                anomaly_map_feature_list = torch.cat(anomaly_map_feature_list, dim=0)

            GT_list = torch.cat(GT_list, dim=0)
            pred_mask = (concat_heatmap > threshold).float()
            visualize(forward_list, reconstructed_list, GT_list, pred_mask, concat_heatmap, config.data.category,
                      config, forward_list_orig, step_list, filename_list, anomaly_map_recon_list,
                      anomaly_map_latent_list, anomaly_map_feature_list)