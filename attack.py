import utils.http_session
import utils.transport
import utils.session
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from scipy import stats
import argparse
import time

BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
SHADOW_DIR = BASE / "shadow_models"
OUTPUT_CSV = BASE / "submission.csv"

MEAN = torch.tensor([0.7406, 0.5331, 0.7059])
STD  = torch.tensor([0.1491, 0.1864, 0.1301])


# ── Dataset wrappers (needed for torch.load deserialisation) ─────────────────

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels, self.transform = [], [], [], transform
    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transform: img = self.transform(img)
        return self.ids[i], img, self.labels[i]
    def __len__(self): return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []
    def __getitem__(self, i):
        id_, img, label = super().__getitem__(i)
        return id_, img, label, self.membership[i]


# ── Model ────────────────────────────────────────────────────────────────────

def create_model():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, 9)
    return m

def load_target_model(device):
    m = create_model()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    m.eval()
    return m.to(device)


# ── Image pre-processing ─────────────────────────────────────────────────────

def preprocess_images(raw_imgs, device):
    """Resize 28→32 and normalise. Returns CPU float32 tensor (N,C,32,32)."""
    if not isinstance(raw_imgs, torch.Tensor):
        raw_imgs = torch.stack(raw_imgs)
    resize = transforms.Resize(32, antialias=True)
    mean = MEAN.view(3, 1, 1)
    std  = STD.view(3, 1, 1)
    chunks = []
    bs = 1024
    for i in range(0, len(raw_imgs), bs):
        b = resize(raw_imgs[i:i+bs].float())
        chunks.append(((b - mean) / std))
    return torch.cat(chunks, dim=0)   # CPU


# ── GPU batch augmentation ────────────────────────────────────────────────────

def augment_batch(imgs, pad=4):
    """Random hflip + reflect-pad + crop. Operates entirely on GPU."""
    B = imgs.shape[0]
    mask = torch.rand(B, device=imgs.device) > 0.5
    imgs = imgs.clone()
    imgs[mask] = imgs[mask].flip(-1)
    padded = F.pad(imgs, [pad]*4, mode='reflect')
    top  = torch.randint(0, 2*pad+1, (1,)).item()
    left = torch.randint(0, 2*pad+1, (1,)).item()
    return padded[:, :, top:top+32, left:left+32]


# ── Shadow model training (GPU-resident, no DataLoader) ───────────────────────

def train_shadow_model(train_imgs, train_labels, device, epochs=60, lr=0.1, batch_size=512):
    model = create_model().to(device)
    imgs_g   = train_imgs.to(device)
    labels_g = train_labels.to(device)
    N = len(imgs_g)

    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            bi  = augment_batch(imgs_g[idx])
            bl  = labels_g[idx]
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = F.cross_entropy(model(bi), bl)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

    model.eval()
    del imgs_g, labels_g
    return model


# ── Logit-margin score ────────────────────────────────────────────────────────

def compute_logit_scores(model, imgs, labels, device, batch_size=2048):
    """logit(correct) - max_logit(others) for every sample."""
    imgs_g   = imgs.to(device)
    labels_g = labels.to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(imgs_g), batch_size):
            bi = imgs_g[i:i+batch_size]
            bl = labels_g[i:i+batch_size]
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(bi).float()
            correct = logits[range(len(bl)), bl]
            rest = logits.clone()
            rest[range(len(bl)), bl] = -1e9
            scores.extend((correct - rest.max(1).values).cpu().tolist())
    del imgs_g, labels_g
    return np.array(scores)


# ── Augmentation-based MIA (no shadow training required) ─────────────────────

def compute_augmentation_features(model, imgs, labels, device,
                                   n_aug=50, batch_size=1024):
    """
    Run n_aug random augmentations of every sample through the (frozen) target
    model and collect per-sample statistics of the logit-margin score.

    Returns dict of arrays of shape (N,): mean, std, min, max, q10, q90.
    Members tend to have higher mean (model is confident on them regardless
    of augmentation) and lower std (model is robust on them).
    """
    imgs_g   = imgs.to(device)
    labels_g = labels.to(device)
    N = len(imgs_g)
    all_scores = np.zeros((n_aug, N), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for a in range(n_aug):
            aug = augment_batch(imgs_g)           # augment entire dataset at once
            scores_a = []
            for i in range(0, N, batch_size):
                bi = aug[i:i+batch_size]
                bl = labels_g[i:i+batch_size]
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(bi).float()
                correct = logits[range(len(bl)), bl]
                rest = logits.clone()
                rest[range(len(bl)), bl] = -1e9
                scores_a.extend((correct - rest.max(1).values).cpu().tolist())
            all_scores[a] = scores_a
            if (a + 1) % 10 == 0:
                print(f"  Aug {a+1}/{n_aug}", flush=True)

    del imgs_g, labels_g
    return {
        'mean': all_scores.mean(0),
        'std':  all_scores.std(0),
        'min':  all_scores.min(0),
        'max':  all_scores.max(0),
        'q10':  np.percentile(all_scores, 10, axis=0),
        'q90':  np.percentile(all_scores, 90, axis=0),
    }


# ── LiRA helpers ──────────────────────────────────────────────────────────────

def _lira_scores(in_scores, out_scores, target_scores, N):
    out = np.empty(N)
    for i in range(N):
        ins  = np.asarray(in_scores[i])
        outs = np.asarray(out_scores[i])
        if len(ins) < 2 or len(outs) < 2:
            out[i] = target_scores[i]
            continue
        li = stats.norm.logpdf(target_scores[i], ins.mean(),  ins.std()  + 1e-10)
        lo = stats.norm.logpdf(target_scores[i], outs.mean(), outs.std() + 1e-10)
        out[i] = li - lo
    return out


def tpr_at_fpr(scores, labels, max_fpr=0.05):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(tpr[np.where(fpr <= max_fpr)[0][-1]])


def normalise_01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-10)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(num_shadow=8, epochs=60, n_aug=50, device_str=None):
    device = torch.device(device_str or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}", flush=True)

    # ── Load raw data ─────────────────────────────────────────────────────────
    print("Loading...", flush=True)
    pub_ds  = torch.load(PUB_PATH,  weights_only=False)
    priv_ds = torch.load(PRIV_PATH, weights_only=False)

    raw_pub  = pub_ds.imgs  if isinstance(pub_ds.imgs,  torch.Tensor) else torch.stack(pub_ds.imgs)
    raw_priv = priv_ds.imgs if isinstance(priv_ds.imgs, torch.Tensor) else torch.stack(priv_ds.imgs)
    pub_labels  = pub_ds.labels  if isinstance(pub_ds.labels,  torch.Tensor) else torch.tensor(pub_ds.labels)
    priv_labels = priv_ds.labels if isinstance(priv_ds.labels, torch.Tensor) else torch.tensor(priv_ds.labels)
    pub_mem = np.array(pub_ds.membership, dtype=int)

    # ── Pre-process ───────────────────────────────────────────────────────────
    print("Pre-processing...", flush=True)
    t0 = time.time()
    pub_imgs  = preprocess_images(raw_pub,  device)
    priv_imgs = preprocess_images(raw_priv, device)
    all_imgs   = torch.cat([pub_imgs,   priv_imgs],   dim=0)   # CPU, (28k,3,32,32)
    all_labels = torch.cat([pub_labels, priv_labels], dim=0)   # CPU
    N, N_pub = len(all_imgs), len(pub_imgs)
    print(f"Done {time.time()-t0:.1f}s  N={N}", flush=True)

    # ── Target model ─────────────────────────────────────────────────────────
    target = load_target_model(device)

    # ── Augmentation-based features ───────────────────────────────────────────
    aug_path = BASE / "aug_features.npz"
    if aug_path.exists():
        print("Loading cached aug features...", flush=True)
        aug_data = np.load(aug_path)
        aug_feats = {k: aug_data[k] for k in aug_data.files}
    else:
        print(f"Computing augmentation features ({n_aug} augs x {N} samples)...", flush=True)
        t0 = time.time()
        aug_feats = compute_augmentation_features(target, all_imgs, all_labels,
                                                   device, n_aug=n_aug)
        np.savez(aug_path, **aug_feats)
        print(f"Aug features done in {time.time()-t0:.1f}s", flush=True)

    # Baseline scores on pub
    aug_score_pub = aug_feats['mean'][:N_pub]
    print(f"Aug-mean  TPR@5%FPR pub: {tpr_at_fpr(aug_score_pub, pub_mem):.4f}", flush=True)
    aug_score2_pub = aug_feats['mean'][:N_pub] - aug_feats['std'][:N_pub]
    print(f"Aug-mean-std TPR@5%FPR pub: {tpr_at_fpr(aug_score2_pub, pub_mem):.4f}", flush=True)

    # ── Target logit scores ───────────────────────────────────────────────────
    print("Target logit scores...", flush=True)
    target_scores = compute_logit_scores(target, all_imgs, all_labels, device)
    print(f"Raw logit  TPR@5%FPR pub: {tpr_at_fpr(target_scores[:N_pub], pub_mem):.4f}", flush=True)

    # ── Reference model (train on pub non-members) ────────────────────────────
    ref_path = BASE / "ref_scores.npz"
    if ref_path.exists():
        print("Loading cached reference model scores...", flush=True)
        ref_scores = np.load(ref_path)['scores']
    else:
        print("Training reference model on pub non-members...", flush=True)
        nm_mask = pub_mem == 0
        ref_imgs   = pub_imgs[nm_mask]
        ref_labels = pub_labels[nm_mask]
        t0 = time.time()
        ref_model = train_shadow_model(ref_imgs, ref_labels, device, epochs=epochs)
        print(f"Ref model took {time.time()-t0:.1f}s", flush=True)
        ref_scores_pub  = compute_logit_scores(ref_model, pub_imgs,  pub_labels,  device)
        ref_scores_priv = compute_logit_scores(ref_model, priv_imgs, priv_labels, device)
        ref_scores = np.concatenate([ref_scores_pub, ref_scores_priv])
        np.savez(ref_path, scores=ref_scores)
        del ref_model
        if device.type == 'cuda': torch.cuda.empty_cache()

    # Reference-model score: higher = more likely member
    ref_signal = ref_scores - target_scores
    print(f"Ref-model  TPR@5%FPR pub: {tpr_at_fpr(ref_signal[:N_pub], pub_mem):.4f}", flush=True)

    # ── Per-class calibrated scores (Zarifzadeh et al.) ──────────────────────
    all_labels_np = all_labels.numpy()

    # Variant 1: subtract non-member class mean (strongest baseline)
    class_mean_nm = np.zeros(9)
    for c in range(9):
        mask_c = (all_labels_np[:N_pub] == c) & (pub_mem == 0)
        class_mean_nm[c] = target_scores[:N_pub][mask_c].mean() if mask_c.sum() > 0 else 0
    per_class_cal = target_scores - class_mean_nm[all_labels_np]

    # Variant 2: per-class rank percentile (rank within class, using pub as reference)
    per_class_rank = np.zeros(N)
    for c in range(9):
        pub_c_mask = all_labels_np[:N_pub] == c
        pub_c_scores = target_scores[:N_pub][pub_c_mask]
        all_c_mask = all_labels_np == c
        # rank each sample against pub class distribution
        per_class_rank[all_c_mask] = np.array([
            (pub_c_scores < target_scores[i]).mean()
            for i in np.where(all_c_mask)[0]
        ])

    # Variant 3: Gaussian LRT using pub member/non-member distributions per class
    from scipy import stats as scipy_stats
    per_class_lrt = np.zeros(N)
    for c in range(9):
        m_mask  = (all_labels_np[:N_pub] == c) & (pub_mem == 1)
        nm_mask = (all_labels_np[:N_pub] == c) & (pub_mem == 0)
        m_s, nm_s = target_scores[:N_pub][m_mask], target_scores[:N_pub][nm_mask]
        if len(m_s) < 2 or len(nm_s) < 2:
            continue
        m_mu,  m_std  = m_s.mean(),  m_s.std()  + 1e-10
        nm_mu, nm_std = nm_s.mean(), nm_s.std() + 1e-10
        c_idx = np.where(all_labels_np == c)[0]
        li = scipy_stats.norm.logpdf(target_scores[c_idx], m_mu,  m_std)
        lo = scipy_stats.norm.logpdf(target_scores[c_idx], nm_mu, nm_std)
        per_class_lrt[c_idx] = li - lo

    print(f"Per-class-cal  TPR@5%FPR pub: {tpr_at_fpr(per_class_cal[:N_pub],  pub_mem):.4f}", flush=True)
    print(f"Per-class-rank TPR@5%FPR pub: {tpr_at_fpr(per_class_rank[:N_pub], pub_mem):.4f}", flush=True)
    print(f"Per-class-LRT  TPR@5%FPR pub: {tpr_at_fpr(per_class_lrt[:N_pub],  pub_mem):.4f}", flush=True)

    # ── LiRA shadow models ────────────────────────────────────────────────────
    SHADOW_DIR.mkdir(exist_ok=True)
    in_scores  = [[] for _ in range(N)]
    out_scores = [[] for _ in range(N)]
    n_train = N // 2

    available_shadows = 0
    for s in range(num_shadow):
        sp = SHADOW_DIR / f"scores_{s}.npz"
        mp = SHADOW_DIR / f"mask_{s}.npy"
        if sp.exists() and mp.exists():
            shadow_scores = np.load(sp)['scores']
            mask = np.load(mp)
        else:
            rng  = np.random.RandomState(s * 31 + 13)
            mask = np.zeros(N, dtype=bool)
            mask[rng.choice(N, n_train, replace=False)] = True
            print(f"Shadow {s}/{num_shadow}: training ({mask.sum()} samples)...", flush=True)
            t0 = time.time()
            shadow = train_shadow_model(all_imgs[mask], all_labels[mask], device, epochs=epochs)
            print(f"  took {time.time()-t0:.1f}s", flush=True)
            shadow_scores = compute_logit_scores(shadow, all_imgs, all_labels, device)
            np.savez(sp, scores=shadow_scores)
            np.save(mp, mask)
            del shadow
            if device.type == 'cuda': torch.cuda.empty_cache()

        for i in range(N):
            (in_scores[i] if mask[i] else out_scores[i]).append(shadow_scores[i])
        available_shadows += 1

        if available_shadows % 2 == 0:
            lira_inter = _lira_scores(in_scores, out_scores, target_scores, N_pub)
            print(f"  [{available_shadows} shadows] LiRA TPR@5%FPR pub: {tpr_at_fpr(lira_inter, pub_mem):.4f}", flush=True)

    lira_full = _lira_scores(in_scores, out_scores, target_scores, N)
    print(f"LiRA ({available_shadows} shadows) TPR@5%FPR pub: {tpr_at_fpr(lira_full[:N_pub], pub_mem):.4f}", flush=True)

    # ── Meta-classifier on pub (uses all signals) ─────────────────────────────
    print("\nTraining meta-classifier on pub features...", flush=True)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    def make_features(idx_slice):
        cols = [
            target_scores[idx_slice],
            aug_feats['mean'][idx_slice],
            aug_feats['std'][idx_slice],
            aug_feats['min'][idx_slice],
            aug_feats['max'][idx_slice],
            aug_feats['q10'][idx_slice],
            aug_feats['q90'][idx_slice],
            ref_signal[idx_slice],
            per_class_cal[idx_slice],
            per_class_rank[idx_slice],
            per_class_lrt[idx_slice],
        ]
        if available_shadows >= 4:
            cols.append(normalise_01(lira_full[idx_slice]))
        return np.column_stack(cols)

    pub_slice  = slice(0, N_pub)
    priv_slice = slice(N_pub, N)

    X_pub  = make_features(pub_slice)
    X_priv = make_features(priv_slice)

    # Cross-validated pub score (avoids overfitting to pub)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    clf_base = Pipeline([
        ('scaler', StandardScaler()),
        ('gbt', GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, subsample=0.8,
                                            random_state=42)),
    ])
    meta_pub_cv = cross_val_predict(clf_base, X_pub, pub_mem,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                     method='predict_proba')[:, 1]
    print(f"Meta-clf CV TPR@5%FPR pub: {tpr_at_fpr(meta_pub_cv, pub_mem):.4f}", flush=True)

    # Refit on all pub data to score priv
    clf_base.fit(X_pub, pub_mem)
    meta_pub_score  = clf_base.predict_proba(X_pub)[:, 1]
    meta_priv_score = clf_base.predict_proba(X_priv)[:, 1]
    print(f"Meta-clf   TPR@5%FPR pub: {tpr_at_fpr(meta_pub_score, pub_mem):.4f}", flush=True)

    # ── Pick best signal to submit ────────────────────────────────────────────
    # Evaluate all candidates on pub, pick best
    from sklearn.metrics import roc_auc_score
    candidates = {
        'aug_mean':       (aug_feats['mean'][:N_pub],     aug_feats['mean'][N_pub:]),
        'ref_signal':     (ref_signal[:N_pub],            ref_signal[N_pub:]),
        'per_class_cal':  (per_class_cal[:N_pub],         per_class_cal[N_pub:]),
        'per_class_rank': (per_class_rank[:N_pub],        per_class_rank[N_pub:]),
        'per_class_lrt':  (per_class_lrt[:N_pub],         per_class_lrt[N_pub:]),
        'lira':           (lira_full[:N_pub],             lira_full[N_pub:]),
        'meta_cv':        (meta_pub_cv,                   meta_priv_score),
        'meta':           (meta_pub_score,                meta_priv_score),
    }
    best_name, best_tpr, best_priv = None, -1, None
    for name, (pub_s, priv_s) in candidates.items():
        t   = tpr_at_fpr(pub_s, pub_mem)
        auc = roc_auc_score(pub_mem, pub_s)
        print(f"  {name:16s} TPR@5%FPR={t:.4f}  AUC={auc:.4f}", flush=True)
        if t > best_tpr:
            best_tpr, best_name, best_priv = t, name, priv_s

    print(f"\nBest signal: {best_name}  (pub TPR@5%FPR = {best_tpr:.4f})", flush=True)

    normed = normalise_01(best_priv)
    pd.DataFrame({"id": [str(i) for i in priv_ds.ids], "score": normed}).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved submission.csv", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_shadow", type=int, default=8)
    p.add_argument("--epochs",     type=int, default=60)
    p.add_argument("--n_aug",      type=int, default=50)
    p.add_argument("--device",     type=str, default=None)
    args = p.parse_args()
    run(num_shadow=args.num_shadow, epochs=args.epochs, n_aug=args.n_aug, device_str=args.device)
