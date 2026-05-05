This repository contains the code used for our best Membership Inference Attack submission.

## Files Used

- `attack.py`: creates `submission.csv`
- `task_template.py`: submits `submission.csv` to the leaderboard
- `pub.pt`: public data with membership labels
- `priv.pt`: private data to score
- `model.pt`: target ResNet-18 model

## Environment

Use a Python environment with:

- PyTorch
- torchvision
- numpy
- pandas
- scipy
- scikit-learn

We ran the attack on a GPU using the provided cluster Docker image:

```bash
pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

## Recreate the Submission

Run the attack from the repository root:

```bash
python attack.py --num_shadow 8 --epochs 60 --n_aug 50
```

This creates:

```bash
submission.csv
```

The script also caches intermediate results:

- `aug_features.npz`
- `ref_scores.npz`
- `shadow_models/`

If these files already exist, rerunning the script will reuse them.

## Submit

After `submission.csv` is created, submit it with:

```bash
export TML_API_KEY="your_api_key_here"
python task_template.py
```

This recreates the submitted leaderboard result.
