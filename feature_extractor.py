import torch
import torch.nn as nn
from tqdm import tqdm
from forward_process import *
from dataset import *
from dataset import *
import timm
import random
from torch import Tensor, nn
from typing import Callable, List, Tuple, Union
from unet import *
from omegaconf import OmegaConf
from sample import *
from visualize import *
from resnet import *
import torchvision.transforms as T
from diffusers import AutoencoderKL


#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

torch.manual_seed(42)

def build_model(config):
    unet = UNetModel(256, 64, dropout=0, n_heads=8 ,in_channels=config.data.fe_input_channel)
    return unet



import torch
import torch.nn as nn

def loss_function(a, b, config):
    """
    結合 Cosine Similarity + L2 Loss（MSE），並用自適應權重控制。
    
    Args:
        a: 預測值 (list or tensor)
        b: 真實值 (list or tensor)
        config: 包含 loss 權重設定的 config

    Returns:
        total_loss: 總損失值（可 backward）
    """
    cos_loss_fn = nn.CosineSimilarity(dim=1, eps=1e-8)  # 加 eps 避免除以 0
    mse_loss_fn = nn.MSELoss()

    # 抓取權重（若無則自動預設值）
    cosine_weight = getattr(config.model, 'cosine_loss_weight', 0.1)
    l2_weight = getattr(config.model, 'l2_loss_weight', 1.0)

    total_loss = 0.0
    total_cosine = 0.0
    total_l2 = 0.0
    count = 0

    # 若為 list（多層輸出），逐層計算
    if isinstance(a, list) and isinstance(b, list):
        for a_i, b_i in zip(a, b):
            a_flat = a_i.view(a_i.shape[0], -1)
            b_flat = b_i.view(b_i.shape[0], -1)

            cos_sim = cos_loss_fn(a_flat, b_flat)  # B x 1
            cosine_loss = torch.mean(1 - cos_sim)

            l2_loss = mse_loss_fn(a_i, b_i)

            total_cosine += cosine_loss.item()
            total_l2 += l2_loss.item()
            count += 1

            total_loss += cosine_weight * cosine_loss + l2_weight * l2_loss
    else:
        a_flat = a.view(a.shape[0], -1)
        b_flat = b.view(b.shape[0], -1)

        cos_sim = cos_loss_fn(a_flat, b_flat)
        cosine_loss = torch.mean(1 - cos_sim)

        l2_loss = mse_loss_fn(a, b)

        total_cosine = cosine_loss.item()
        total_l2 = l2_loss.item()
        total_loss = cosine_weight * cosine_loss + l2_weight * l2_loss

    # debug 顯示（可選）
    if getattr(config.model, 'debug', False):
        if count > 0:
            avg_cosine = total_cosine / count
            avg_l2 = total_l2 / count
        else:
            avg_cosine = total_cosine
            avg_l2 = total_l2
        print(f"[CustomLoss] Total: {total_loss.item():.6f} | Cosine: {avg_cosine:.6f} | L2: {avg_l2:.6f}")

    return total_loss



def roundup(x, n=10):
    res = math.ceil(x/n)*n
    if (x%n < n/2)and (x%n>0):
        res-=n
    return res
              

def Domain_adaptation(unet, feature_extractor, vae, config, fine_tune, constants_dict, dataloader, consistency_decoder):
    if fine_tune:
        unet.eval()
        feature_extractor.train()
        for param in feature_extractor.parameters():
            param.requires_grad = True

        transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        optimizer = torch.optim.AdamW(feature_extractor.parameters(), lr=config.model.DA_learning_rate)

        for epoch in tqdm(range(config.model.DA_epochs), desc="Epoch Progress"):
            dataloader_tqdm = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)

            for step, batch in enumerate(dataloader_tqdm):
                with torch.no_grad():
                    if config.model.DA_rnd_step:
                        step_percentage = np.random.randint(1, 11)
                        step_size = config.model.test_trajectoy_steps_DA / 10 * step_percentage
                        test_trajectoy_steps = torch.Tensor([step_size]).type(torch.int64).to(config.model.device)
                        step_size = roundup(step_size)
                        skip = int(max(step_size / 10, 1))
                        seq = range(0, step_size, skip)
                    else:
                        test_trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps_DA]).type(torch.int64).to(config.model.device)
                        seq = range(0, config.model.test_trajectoy_steps_DA, config.model.skip_DA)

                    at = compute_alpha(constants_dict["betas"], test_trajectoy_steps.long(), config)

                    target = batch[0].to(config.model.device)
                    target_vae = vae.encode(target).latent_dist.sample() * 0.18215

                    if config.model.noise_sampling:
                        noise = torch.randn_like(target_vae).to(config.model.device)
                        noisy_image = at.sqrt() * target_vae + (1 - at).sqrt() * noise
                    else:
                        noisy_image = target_vae
                        if config.model.downscale_first:
                            noisy_image = noisy_image * at.sqrt()

                    reconstructed, _ = DA_generalized_steps(
                        target_vae, noisy_image, seq, unet, constants_dict["betas"], config,
                        eta2=config.model.eta2, eta3=0, constants_dict=constants_dict, eraly_stop=False
                    )

                    data_reconstructed = reconstructed[-1].to(config.model.device)
                    del target_vae, noisy_image
                    torch.cuda.empty_cache()

                    data_reconstructed = 1 / 0.18215 * data_reconstructed
                    if config.model.consistency_decoder:
                        data_reconstructed = consistency_decoder(data_reconstructed)
                    else:
                        data_reconstructed = vae.decode(data_reconstructed).sample

                data_reconstructed = transform(data_reconstructed)
                reconst_fe = feature_extractor(data_reconstructed)
                del data_reconstructed
                torch.cuda.empty_cache()

                target = transform(target)
                target_fe = feature_extractor(target)

                loss = loss_function(reconst_fe, target_fe, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                dataloader_tqdm.set_postfix(loss=loss.item())

            print(f"[Epoch {epoch+1}/{config.model.DA_epochs}] Loss: {loss.item():.6f}")

            ckpt_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                feature_extractor.state_dict(),
                os.path.join(ckpt_dir, f'feature_recon_sim{epoch + 1}')
            )

    else:
        checkpoint_path = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, f'feature_recon_sim{config.model.DA_epochs}')
        feature_extractor.load_state_dict(torch.load(checkpoint_path))
        print("Loaded feature extractor from checkpoint.")

    return feature_extractor