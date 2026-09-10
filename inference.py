"""RTMDiff-Spatial inference for 256 x 256 input patches."""

import os
import glob
import json
import numpy as np
import torch
import torch.utils.data as tud
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.schedulers import DDPMScheduler
from models import ConditionalUNet


# Paths are relative to the current working directory.
data_dir_input = './input/'
data_dir = './'
model_file = './diffussion_model.pth'
output_path = './output/'

DEVICE = 'cuda:0'
BATCH_SIZE = 24
NUM_SAMPLING_STEPS = 800
ENSEMBLE_SIZE = 30
SAVE_MEMBERS = False

# Input: 13 channels, each with shape (256, 256).
input_variables = [
    'cos_saz', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15',
    'bc10', 'bc11', 'bc12', 'bc13', 'bc14', 'bc15',
]
label_variables = ['cth', 'log_cot', 'cer']

with open(os.path.join(data_dir, 'config.json'), 'r') as config_file:
    config_data = json.load(config_file)
input_para = np.array([
    [config_data[var]['mean'][0] for var in input_variables],
    [config_data[var]['std'][0] for var in input_variables],
])

with open(os.path.join(data_dir, 'config_label.json'), 'r') as config_file:
    config_data = json.load(config_file)
label_para = np.array([
    [config_data[var]['mean'][0] for var in label_variables],
    [config_data[var]['std'][0] for var in label_variables],
])


class CustomDataset(tud.Dataset):
    def __init__(self):
        self.input_files = self.load_data_files(data_dir_input)
        if not self.input_files:
            raise FileNotFoundError(f'No input .npy files found in {data_dir_input}')

    def load_data_files(self, directory):
        data_files = sorted(glob.glob(os.path.join(directory, '*.npy')))
        # Keep the original ordering for DOY/time/patch filenames.
        try:
            data_files = sorted(data_files, key=lambda x: (
                int(os.path.basename(x)[-16:-13]),
                int(os.path.basename(x)[-12:-8]),
                int(os.path.basename(x)[-7:-4]),
            ))
        except ValueError:
            pass  # Other filenames retain alphabetical ordering.
        return data_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_file = self.input_files[index]
        # Original np.save(file, dictionary) format; use trusted input files.
        input_data_dict = np.load(input_file, allow_pickle=True).item()

        for var in input_variables:
            if var not in input_data_dict:
                raise KeyError(f'{input_file}: missing input variable {var}')
            if input_data_dict[var].ndim == 2:
                input_data_dict[var] = input_data_dict[var][np.newaxis, :, :]
            if input_data_dict[var].shape != (1, 256, 256):
                raise ValueError(f'{input_file}: {var} must have shape (256, 256) or (1, 256, 256)')
            if not np.issubdtype(input_data_dict[var].dtype, np.floating):
                raise TypeError(f'{input_file}: {var} must be a floating-point array')

        input_data = np.concatenate(
            [input_data_dict[var] for var in input_variables], axis=0
        )
        input_data = preprocess(input_data)
        return input_data, input_file


def preprocess(input_data):
    for i in range(input_data.shape[0]):
        input_data[i, :, :] = (
            input_data[i, :, :] - input_para[0, i]
        ) / input_para[1, i]
    return input_data


def postprocess_gpu(sampled_var):
    sampled_output = sampled_var.clone()
    for i in range(sampled_output.shape[1]):
        if i == 1:
            sampled_output[:, i, :, :] = torch.exp(
                sampled_output[:, i, :, :] * label_para[1, i] + label_para[0, i]
            )
        else:
            sampled_output[:, i, :, :] = (
                sampled_output[:, i, :, :] * label_para[1, i] + label_para[0, i]
            )
    return sampled_output


def create_data_loader(batch_size=BATCH_SIZE, shuffle=True,
                       drop_last=False, num_workers=0):
    custom_ds = CustomDataset()
    dl = tud.DataLoader(
        custom_ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers, pin_memory=False,
    )
    return dl


def make_conditions(timesteps: torch.Tensor, images: torch.Tensor,
                    embedding_dim: int = 13) -> torch.Tensor:
    timestep_embedding = get_timestep_embedding(
        timesteps, embedding_dim, max_period=10000
    )
    timestep_embedding = timestep_embedding[:, None, :]
    img_embed = torch.flatten(images, 2)
    img_embed = torch.swapdims(img_embed, 1, 2)
    return torch.cat([timestep_embedding, img_embed], dim=1)


def sampling_diffusion():
    if not 1 <= NUM_SAMPLING_STEPS <= 800:
        raise ValueError('NUM_SAMPLING_STEPS must be between 1 and 800')
    if ENSEMBLE_SIZE < 2:
        raise ValueError('ENSEMBLE_SIZE must be at least 2 to calculate standard deviation')
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f'Model weights not found: {model_file}')

    default_device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    default_type = torch.float32
    print('Device:', default_device)

    unet = ConditionalUNet(3, condition_dim=13)
    unet.load_state_dict(torch.load(model_file, map_location='cpu'), strict=True)
    unet.to(device=default_device, dtype=default_type)
    unet.requires_grad_(False)
    unet.eval()

    dl = create_data_loader(shuffle=True)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=800,
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type='epsilon',
        clip_sample=False,
    )
    noise_scheduler.set_timesteps(NUM_SAMPLING_STEPS)
    timesteps = noise_scheduler.timesteps
    os.makedirs(output_path, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (xx, input_files) in enumerate(dl):
            xx = xx.to(device=default_device, dtype=default_type)
            xx = torch.where(
                torch.isnan(xx),
                torch.tensor(1.0, device=default_device, dtype=default_type), xx,
            )
            samples = []
            for i in range(ENSEMBLE_SIZE):
                print(f'Batch {batch_idx + 1}/{len(dl)}, member {i + 1}/{ENSEMBLE_SIZE}')
                tgt_latent = torch.randn(
                    [xx.shape[0], 3, 256, 256],
                    device=default_device, dtype=default_type,
                )
                for t in timesteps.tolist():
                    ts = torch.full(
                        [xx.shape[0]], t, device=default_device, dtype=default_type
                    )
                    conditions = make_conditions(ts, xx, embedding_dim=13)
                    conditional_noise = unet(tgt_latent, conditions)
                    tgt_latent = noise_scheduler.step(
                        conditional_noise, t, tgt_latent
                    ).prev_sample
                samples.append(postprocess_gpu(tgt_latent))

            samples = torch.stack(samples)
            tgt_mean = torch.mean(samples, dim=0)
            tgt_std = torch.std(samples, dim=0)

            for i in range(tgt_mean.shape[0]):
                base_name = os.path.splitext(os.path.basename(input_files[i]))[0]
                np.save(
                    os.path.join(output_path, f'{base_name}_tgt1.npy'),
                    tgt_mean[i].cpu().numpy().astype(np.float32),
                )
                np.save(
                    os.path.join(output_path, f'{base_name}_tgt_std1.npy'),
                    tgt_std[i].cpu().numpy().astype(np.float32),
                )
                if SAVE_MEMBERS:
                    np.save(
                        os.path.join(output_path, f'{base_name}_tgt_samples1.npy'),
                        samples[:, i].cpu().numpy().astype(np.float32),
                    )
                print('Saved:', base_name)


def main():
    sampling_diffusion()


if __name__ == '__main__':
    main()
