# RTMDiff-Spatial

Retrieve cloud-top height (CTH), cloud optical thickness (COT), and cloud effective radius (CER) from FY-4B AGRI thermal infrared observations.

## Files

```text
inference.py           # Inference script
models.py              # Conditional U-Net
config.json            # Input normalization
config_label.json      # Output inverse normalization
diffussion_model.pth   # Trained model weights; available from Releases
requirements.txt
input/                 # Input .npy files
output/                # Saved results; created automatically
```
## Model weights

The trained model weights (`diffussion_model.pth`) are available from the GitHub Releases page.

## Input

Each `.npy` file contains a dictionary of 13 floating-point arrays. Each array has shape **(256, 256)** or **(1, 256, 256)**, giving an input patch of **(13, 256, 256)**.

| Dictionary keys | Input | Units |
|---|---|---|
| `cos_saz` | Cosine of the satellite viewing zenith angle | Dimensionless |
| `b10`–`b15` | Observed AGRI B10–B15 brightness temperatures | K |
| `bc10`–`bc15` | Corresponding simulated clear-sky brightness temperatures | K |

Supply values in the units above; normalization is applied by the script. Keep the supplied JSON statistics paired with the model weights.

## Run

From the directory containing the scripts, JSON files, and weights:

```bash
python3 -m pip install -r requirements.txt
python3 inference.py
```

Paths and settings are at the top of `inference.py`. Defaults are `cuda:0` (CPU fallback), batch size 24, **800 sampling steps**, and **30 ensemble members**.

## Output

For `scene_001.npy`, the script saves:

```text
scene_001_tgt1.npy       # Ensemble mean: (3, 256, 256)
scene_001_tgt_std1.npy   # Absolute ensemble standard deviation: (3, 256, 256)
```

Both files use channel order **CTH (km), COT (dimensionless), CER (µm)**. COT is saved in physical rather than logarithmic units.

Set `SAVE_MEMBERS = True` to also save `scene_001_tgt_samples1.npy` with shape **(30, 3, 256, 256)**.

For faster sampling, `NUM_SAMPLING_STEPS` can be reduced, for example to 400. This can change the results; retain 800 for the original sampling configuration.
