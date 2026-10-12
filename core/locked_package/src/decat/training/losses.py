"""Physics-informed and group-aware training losses."""

from __future__ import annotations
from contextlib import nullcontext
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
import decat.chemistry as _chemistry
import decat.config as _config
import decat.data.dataset as _data_dataset


def _build_category_physics_meta(
    dataset: _data_dataset.FingerprintReactionDataset, train_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    cat_mat = np.asarray(dataset.category, dtype=np.float32)
    if cat_mat.ndim != 2 or cat_mat.shape[1] <= 0:
        zeros = np.zeros((1,), dtype=np.float32)
        return {
            "sign": zeros.copy(),
            "target_slope": zeros.copy(),
            "confidence": zeros.copy(),
            "ph_lo": np.asarray([-1.0], dtype=np.float32),
            "ph_hi": np.asarray([1.0], dtype=np.float32),
            "slope": zeros.copy(),
            "corr": zeros.copy(),
            "valid": zeros.copy(),
        }
    n_cat = int(cat_mat.shape[1])
    cat_idx_all = np.argmax(cat_mat, axis=1).astype(int)
    n_samples = int(cat_idx_all.shape[0])
    ph_raw = np.asarray(dataset.ph, dtype=float).reshape(-1)
    ph_scaled = np.asarray(
        getattr(dataset, "ph_scaled", dataset.ph), dtype=float
    ).reshape(-1)
    y_raw = np.asarray(dataset.logk_raw, dtype=float).reshape(-1)
    idx = np.asarray(train_idx, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < cat_idx_all.shape[0])]
    min_samples = int(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_MIN_SAMPLES",
            "DECAT_RUNTIME_PHYSICS_MIN_SAMPLES_FALLBACK",
            "18",
        )
    )
    corr_thr = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_CORR_THR",
            "DECAT_RUNTIME_PHYSICS_CORR_THR_FALLBACK",
            "0.12",
        )
    )
    slope_thr = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_SLOPE_THR",
            "DECAT_RUNTIME_PHYSICS_SLOPE_THR_FALLBACK",
            "0.015",
        )
    )
    span_thr = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_SPAN_THR",
            "DECAT_RUNTIME_PHYSICS_SPAN_THR_FALLBACK",
            "0.35",
        )
    )
    conf_span_ref = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_CONF_SPAN_REF",
            "DECAT_RUNTIME_PHYSICS_CONF_SPAN_REF_FALLBACK",
            "2.00",
        )
    )
    within_smiles_prior = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_WITHIN_SMILES",
        "DECAT_RUNTIME_PHYSICS_WITHIN_SMILES_FALLBACK",
        False,
    )
    within_min_groups = int(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_WITHIN_MIN_GROUPS",
            "DECAT_RUNTIME_PHYSICS_WITHIN_MIN_GROUPS_FALLBACK",
            "5",
        )
    )
    within_min_agreement = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_WITHIN_MIN_AGREEMENT",
            "DECAT_RUNTIME_PHYSICS_WITHIN_MIN_AGREEMENT_FALLBACK",
            "0.65",
        )
    )
    fg_state_prior = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_FG_STATE",
        "DECAT_RUNTIME_PHYSICS_FG_STATE_FALLBACK",
        False,
    )
    fg_state_width = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_FG_WIDTH",
            "DECAT_RUNTIME_PHYSICS_FG_WIDTH_FALLBACK",
            "1.35",
        )
    )
    fg_state_conf = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_FG_CONF",
            "DECAT_RUNTIME_PHYSICS_FG_CONF_FALLBACK",
            "0.38",
        )
    )
    sign_prior = np.zeros((n_cat,), dtype=np.float32)
    target_slope_prior = np.zeros((n_cat,), dtype=np.float32)
    conf_prior = np.zeros((n_cat,), dtype=np.float32)
    slope_prior = np.zeros((n_cat,), dtype=np.float32)
    corr_prior = np.zeros((n_cat,), dtype=np.float32)
    valid_prior = np.zeros((n_cat,), dtype=np.float32)
    ph_lo = np.full((n_cat,), -1.0, dtype=np.float32)
    ph_hi = np.full((n_cat,), 1.0, dtype=np.float32)
    sample_sign = np.zeros((n_samples,), dtype=np.float32)
    sample_target_slope = np.zeros((n_samples,), dtype=np.float32)
    sample_conf = np.zeros((n_samples,), dtype=np.float32)
    sample_valid = np.zeros((n_samples,), dtype=np.float32)
    sample_ph_lo = np.full((n_samples,), -1.0, dtype=np.float32)
    sample_ph_hi = np.full((n_samples,), 1.0, dtype=np.float32)
    if idx.size == 0:
        return {
            "sign": sign_prior,
            "target_slope": target_slope_prior,
            "confidence": conf_prior,
            "ph_lo": ph_lo,
            "ph_hi": ph_hi,
            "slope": slope_prior,
            "corr": corr_prior,
            "valid": valid_prior,
            "sample_sign": sample_sign,
            "sample_target_slope": sample_target_slope,
            "sample_confidence": sample_conf,
            "sample_ph_lo": sample_ph_lo,
            "sample_ph_hi": sample_ph_hi,
            "sample_valid": sample_valid,
        }
    if ph_scaled[idx].size > 0:
        global_lo = float(np.quantile(ph_scaled[idx], 0.05))
        global_hi = float(np.quantile(ph_scaled[idx], 0.95))
        ph_lo[:] = global_lo
        ph_hi[:] = max(global_hi, global_lo + 0.001)
    ph_scale = 1.0
    y_scale = 1.0
    try:
        ph_scale = float(
            np.asarray(dataset.ph_scaler.scale_, dtype=float).reshape(-1)[0]
        )
    except Exception:
        ph_scale = 1.0
    try:
        y_scale = float(
            np.asarray(dataset.logk_scaler.scale_, dtype=float).reshape(-1)[0]
        )
    except Exception:
        y_scale = 1.0
    if not np.isfinite(ph_scale) or abs(ph_scale) <= 1e-12:
        ph_scale = 1.0
    if not np.isfinite(y_scale) or abs(y_scale) <= 1e-12:
        y_scale = 1.0
    use_sample_smiles_prior = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_SAMPLE_SMILES",
        "DECAT_RUNTIME_PHYSICS_SAMPLE_SMILES_FALLBACK",
        within_smiles_prior,
    )
    sample_local_slope = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_SLOPE",
        "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_SLOPE_FALLBACK",
        False,
    )
    sample_local_bw = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_BW",
            "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_BW_FALLBACK",
            "1.20",
        )
    )
    sample_local_blend = float(
        _config._env_value(
            "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_BLEND",
            "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_BLEND_FALLBACK",
            "0.65",
        )
    )
    sample_local_guard = _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_SIGN_GUARD",
        "DECAT_RUNTIME_PHYSICS_SAMPLE_LOCAL_SIGN_GUARD_FALLBACK",
        True,
    )
    if use_sample_smiles_prior:
        smiles_all = np.asarray(
            getattr(dataset, "smiles", np.asarray([""] * n_samples)), dtype=object
        )
        grouped_all: Dict[str, list] = {}
        for row_idx in idx.tolist():
            grouped_all.setdefault(str(smiles_all[int(row_idx)]), []).append(
                int(row_idx)
            )
        for group_indices in grouped_all.values():
            if len(group_indices) < 2:
                continue
            gx = ph_raw[group_indices].astype(float)
            gy = y_raw[group_indices].astype(float)
            gspan = float(np.max(gx) - np.min(gx)) if gx.size > 0 else 0.0
            if gx.size < 2 or gspan < max(0.1, 0.5 * span_thr):
                continue
            gx0 = gx - float(np.mean(gx))
            gy0 = gy - float(np.mean(gy))
            den = float(np.sum(gx0 * gx0))
            if den <= 1e-10:
                continue
            gs = float(np.sum(gx0 * gy0) / (den + 1e-12))
            if not np.isfinite(gs) or abs(gs) < slope_thr:
                continue
            scaled_slope = float(gs * ph_scale / y_scale)
            gidx = np.asarray(group_indices, dtype=int)
            sample_sign[gidx] = 1.0 if gs > 0.0 else -1.0
            sample_target_slope[gidx] = scaled_slope
            sample_valid[gidx] = 1.0
            support_score = min(1.0, float(len(group_indices)) / 4.0)
            span_score = min(
                1.0,
                max(
                    0.0,
                    (gspan - span_thr * 0.5)
                    / max(1e-06, conf_span_ref - span_thr * 0.5),
                ),
            )
            slope_score = min(
                1.0, max(0.0, (abs(gs) - slope_thr) / max(1e-06, 4.0 * slope_thr))
            )
            base_conf = float(
                0.4 * support_score + 0.35 * span_score + 0.25 * slope_score
            )
            sample_conf[gidx] = base_conf
            gscaled = ph_scaled[gidx].astype(float)
            if gscaled.size > 0:
                lo = float(np.min(gscaled))
                hi = float(np.max(gscaled))
                if np.isfinite(lo) and np.isfinite(hi):
                    sample_ph_lo[gidx] = lo
                    sample_ph_hi[gidx] = max(hi, lo + 0.001)
            if sample_local_slope:
                bw = max(0.15, sample_local_bw)
                blend = float(max(0.0, min(1.0, sample_local_blend)))
                for local_pos, row_idx in enumerate(group_indices):
                    center = float(gx[local_pos])
                    dist = gx - center
                    weights = np.exp(-0.5 * (dist / bw) ** 2).astype(float)
                    if weights.size <= 0 or float(np.sum(weights)) <= 1e-10:
                        continue
                    lx0 = gx - float(
                        np.sum(weights * gx) / max(1e-12, float(np.sum(weights)))
                    )
                    ly0 = gy - float(
                        np.sum(weights * gy) / max(1e-12, float(np.sum(weights)))
                    )
                    lden = float(np.sum(weights * lx0 * lx0))
                    local_slope = gs
                    if lden > 1e-10:
                        candidate_slope = float(
                            np.sum(weights * lx0 * ly0) / (lden + 1e-12)
                        )
                        if np.isfinite(candidate_slope):
                            if sample_local_guard and candidate_slope * gs < 0.0:
                                local_slope = gs
                            else:
                                local_slope = (
                                    1.0 - blend
                                ) * gs + blend * candidate_slope
                    if not np.isfinite(local_slope) or abs(local_slope) < slope_thr:
                        local_slope = gs
                    row_idx = int(row_idx)
                    scaled_local = float(local_slope * ph_scale / y_scale)
                    sample_sign[row_idx] = 1.0 if local_slope > 0.0 else -1.0
                    sample_target_slope[row_idx] = scaled_local
                    neff = float(
                        np.sum(weights) ** 2 / max(1e-12, np.sum(weights * weights))
                    )
                    local_support_score = min(1.0, neff / 3.0)
                    sample_conf[row_idx] = float(
                        max(0.05, base_conf * (0.7 + 0.3 * local_support_score))
                    )
                    try:
                        lo_scaled = float(
                            dataset.ph_scaler.transform(
                                np.asarray([[center - 1.5 * bw]], dtype=float)
                            )[0, 0]
                        )
                        hi_scaled = float(
                            dataset.ph_scaler.transform(
                                np.asarray([[center + 1.5 * bw]], dtype=float)
                            )[0, 0]
                        )
                    except Exception:
                        lo_scaled, hi_scaled = (
                            sample_ph_lo[row_idx],
                            sample_ph_hi[row_idx],
                        )
                    if np.isfinite(lo_scaled) and np.isfinite(hi_scaled):
                        sample_ph_lo[row_idx] = min(lo_scaled, hi_scaled)
                        sample_ph_hi[row_idx] = max(
                            max(lo_scaled, hi_scaled), min(lo_scaled, hi_scaled) + 0.001
                        )
    if _config._env_bool_pair(
        "DECAT_RUNTIME_PHYSICS_KNN_PROPAGATE",
        "DECAT_RUNTIME_PHYSICS_KNN_PROPAGATE_FALLBACK",
        False,
    ):
        train_idx_arr = np.asarray(idx, dtype=int)
        anchor_idx = train_idx_arr[sample_valid[train_idx_arr] > 0.5]
        target_idx = train_idx_arr[sample_valid[train_idx_arr] <= 0.5]
        if anchor_idx.size > 0 and target_idx.size > 0:
            fp_all = np.asarray(
                getattr(
                    dataset, "fingerprint", np.zeros((n_samples, 0), dtype=np.float32)
                ),
                dtype=np.float32,
            )
            if (
                fp_all.ndim == 2
                and fp_all.shape[0] == n_samples
                and (fp_all.shape[1] > 0)
            ):
                fp_bool = fp_all > 0.5
                anchor_fp = fp_bool[anchor_idx]
                anchor_sum = anchor_fp.sum(axis=1).astype(np.float32)
                anchor_slope = sample_target_slope[anchor_idx].astype(np.float32)
                anchor_sign = sample_sign[anchor_idx].astype(np.float32)
                anchor_conf = sample_conf[anchor_idx].astype(np.float32)
                anchor_cat = cat_idx_all[anchor_idx]
                k_nn = int(
                    _config._env_value(
                        "DECAT_RUNTIME_PHYSICS_KNN_K",
                        "DECAT_RUNTIME_PHYSICS_KNN_K_FALLBACK",
                        "7",
                    )
                )
                sim_thr = float(
                    _config._env_value(
                        "DECAT_RUNTIME_PHYSICS_KNN_SIM_THR",
                        "DECAT_RUNTIME_PHYSICS_KNN_SIM_THR_FALLBACK",
                        "0.22",
                    )
                )
                agreement_thr = float(
                    _config._env_value(
                        "DECAT_RUNTIME_PHYSICS_KNN_AGREE_THR",
                        "DECAT_RUNTIME_PHYSICS_KNN_AGREE_THR_FALLBACK",
                        "0.65",
                    )
                )
                same_cat_only = _config._env_bool_pair(
                    "DECAT_RUNTIME_PHYSICS_KNN_SAME_CAT",
                    "DECAT_RUNTIME_PHYSICS_KNN_SAME_CAT_FALLBACK",
                    True,
                )
                for row_idx in target_idx.tolist():
                    x_fp = fp_bool[int(row_idx)]
                    x_sum = float(np.sum(x_fp))
                    if x_sum <= 0.0:
                        continue
                    inter = (
                        np.logical_and(anchor_fp, x_fp).sum(axis=1).astype(np.float32)
                    )
                    union = anchor_sum + x_sum - inter
                    sim = inter / np.maximum(union, 1.0)
                    if same_cat_only:
                        sim = np.where(
                            anchor_cat == int(cat_idx_all[int(row_idx)]), sim, 0.0
                        )
                    valid_neighbor = sim >= sim_thr
                    if not np.any(valid_neighbor):
                        continue
                    cand = np.where(valid_neighbor)[0]
                    order = cand[np.argsort(sim[cand])[::-1]]
                    order = order[: max(1, k_nn)]
                    if order.size <= 0:
                        continue
                    signs = anchor_sign[order]
                    weights = sim[order] * np.clip(anchor_conf[order], 0.001, 1.0)
                    if float(np.sum(weights)) <= 1e-08:
                        continue
                    pos_w = float(np.sum(weights[signs > 0.0]))
                    neg_w = float(np.sum(weights[signs < 0.0]))
                    total_w = pos_w + neg_w
                    if total_w <= 1e-08:
                        continue
                    agreement = max(pos_w, neg_w) / total_w
                    if agreement < agreement_thr:
                        continue
                    slope = float(
                        np.sum(weights * anchor_slope[order]) / np.sum(weights)
                    )
                    if not np.isfinite(slope) or abs(slope) < 1e-05:
                        continue
                    sample_target_slope[int(row_idx)] = slope
                    sample_sign[int(row_idx)] = 1.0 if slope > 0.0 else -1.0
                    sample_valid[int(row_idx)] = 1.0
                    sim_score = float(np.max(sim[order]))
                    conf_score = float(np.sum(weights) / max(1e-06, np.sum(sim[order])))
                    sample_conf[int(row_idx)] = float(
                        min(
                            0.85,
                            0.45 * agreement
                            + 0.35 * min(1.0, sim_score / max(sim_thr, 1e-06))
                            + 0.2 * min(1.0, conf_score),
                        )
                    )
                    src = anchor_idx[order]
                    lo = float(np.min(sample_ph_lo[src]))
                    hi = float(np.max(sample_ph_hi[src]))
                    if np.isfinite(lo) and np.isfinite(hi):
                        sample_ph_lo[int(row_idx)] = lo
                        sample_ph_hi[int(row_idx)] = max(hi, lo + 0.001)
    if fg_state_prior:
        smiles_all = np.asarray(
            getattr(dataset, "smiles", np.asarray([""] * n_samples)), dtype=object
        )
        fg_width = float(max(0.25, fg_state_width))
        fg_base_conf = float(max(0.05, min(0.95, fg_state_conf)))
        for row_idx in idx.tolist():
            row_idx = int(row_idx)
            if sample_valid[row_idx] > 0.5:
                continue
            hits = _chemistry._ionizable_physics_hits(str(smiles_all[row_idx]))
            if not hits:
                continue
            ph_val = float(ph_raw[row_idx]) if np.isfinite(ph_raw[row_idx]) else 7.0
            scored = []
            for rule in hits:
                pka = float(rule.get("pka", 7.0))
                window = float(np.exp(-0.5 * ((ph_val - pka) / fg_width) ** 2))
                if window < 0.08:
                    continue
                raw_slope = float(rule.get("slope", 0.06)) * float(
                    rule.get("sign", 1.0)
                )
                scored.append((window, raw_slope, pka))
            if not scored:
                continue
            weights = np.asarray([s[0] for s in scored], dtype=float)
            raw_slopes = np.asarray([s[1] for s in scored], dtype=float)
            pka_vals = np.asarray([s[2] for s in scored], dtype=float)
            raw_slope = float(
                np.sum(weights * raw_slopes) / max(1e-08, float(np.sum(weights)))
            )
            if not np.isfinite(raw_slope) or abs(raw_slope) < 1e-06:
                continue
            scaled_slope = float(raw_slope * ph_scale / y_scale)
            sample_sign[row_idx] = 1.0 if scaled_slope > 0.0 else -1.0
            sample_target_slope[row_idx] = scaled_slope
            sample_valid[row_idx] = 1.0
            sample_conf[row_idx] = float(
                min(0.8, fg_base_conf * min(1.0, float(np.max(weights))))
            )
            pka_center = float(
                np.sum(weights * pka_vals) / max(1e-08, float(np.sum(weights)))
            )
            lo_raw = pka_center - 1.75 * fg_width
            hi_raw = pka_center + 1.75 * fg_width
            try:
                lo_scaled = float(
                    dataset.ph_scaler.transform(np.asarray([[lo_raw]], dtype=float))[
                        0, 0
                    ]
                )
                hi_scaled = float(
                    dataset.ph_scaler.transform(np.asarray([[hi_raw]], dtype=float))[
                        0, 0
                    ]
                )
            except Exception:
                lo_scaled, hi_scaled = (-1.0, 1.0)
            sample_ph_lo[row_idx] = min(lo_scaled, hi_scaled)
            sample_ph_hi[row_idx] = max(
                max(lo_scaled, hi_scaled), min(lo_scaled, hi_scaled) + 0.001
            )
    for c in range(n_cat):
        c_idx = idx[cat_idx_all[idx] == c]
        if c_idx.size < min_samples:
            continue
        x = ph_raw[c_idx].astype(float)
        y = y_raw[c_idx].astype(float)
        x0 = x - float(np.mean(x))
        y0 = y - float(np.mean(y))
        var_x = float(np.sum(x0 * x0))
        if var_x <= 1e-10:
            continue
        slope = float(np.sum(x0 * y0) / (var_x + 1e-12))
        try:
            corr = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            corr = 0.0
        slope_prior[c] = float(slope) if np.isfinite(slope) else 0.0
        corr_prior[c] = float(corr) if np.isfinite(corr) else 0.0
        if np.isfinite(slope):
            target_slope_prior[c] = float(slope * ph_scale / y_scale)
        x_scaled = ph_scaled[c_idx].astype(float)
        if x_scaled.size > 0:
            lo = float(np.quantile(x_scaled, 0.05))
            hi = float(np.quantile(x_scaled, 0.95))
            if np.isfinite(lo) and np.isfinite(hi):
                ph_lo[c] = lo
                ph_hi[c] = max(hi, lo + 0.001)
        ph_span = float(np.max(x) - np.min(x)) if x.size > 0 else 0.0
        if within_smiles_prior:
            grouped: Dict[str, list] = {}
            smiles_all = np.asarray(
                getattr(dataset, "smiles", np.asarray([""] * cat_idx_all.shape[0])),
                dtype=object,
            )
            for row_idx in c_idx.tolist():
                grouped.setdefault(str(smiles_all[int(row_idx)]), []).append(
                    int(row_idx)
                )
            within_slopes = []
            for group_indices in grouped.values():
                if len(group_indices) < 2:
                    continue
                gx = ph_raw[group_indices].astype(float)
                gy = y_raw[group_indices].astype(float)
                if gx.size < 2 or float(np.max(gx) - np.min(gx)) < max(
                    0.1, 0.5 * span_thr
                ):
                    continue
                gx0 = gx - float(np.mean(gx))
                gy0 = gy - float(np.mean(gy))
                den = float(np.sum(gx0 * gx0))
                if den <= 1e-10:
                    continue
                gs = float(np.sum(gx0 * gy0) / (den + 1e-12))
                if np.isfinite(gs):
                    within_slopes.append(gs)
            if len(within_slopes) >= max(1, within_min_groups):
                ws = np.asarray(within_slopes, dtype=float)
                med_slope = float(np.median(ws))
                pos_frac = float(np.mean(ws > 0.0))
                neg_frac = float(np.mean(ws < 0.0))
                agreement = max(pos_frac, neg_frac)
                if abs(med_slope) >= slope_thr and agreement >= within_min_agreement:
                    sign_prior[c] = 1.0 if med_slope > 0.0 else -1.0
                    slope_prior[c] = med_slope
                    target_slope_prior[c] = float(med_slope * ph_scale / y_scale)
                    corr_prior[c] = float(2.0 * agreement - 1.0) * float(sign_prior[c])
                    support_score = min(
                        1.0, float(len(ws)) / float(max(within_min_groups * 3, 1))
                    )
                    agreement_score = min(
                        1.0,
                        max(
                            0.0,
                            (agreement - within_min_agreement)
                            / max(1e-06, 1.0 - within_min_agreement),
                        ),
                    )
                    slope_score = min(
                        1.0,
                        max(
                            0.0,
                            (abs(med_slope) - slope_thr) / max(1e-06, 4.0 * slope_thr),
                        ),
                    )
                    conf_prior[c] = float(
                        0.45 * agreement_score
                        + 0.35 * support_score
                        + 0.2 * slope_score
                    )
                    valid_prior[c] = 1.0
                    continue
        if (
            np.isfinite(slope)
            and np.isfinite(corr)
            and (abs(corr) >= corr_thr)
            and (abs(slope) >= slope_thr)
            and (ph_span >= span_thr)
        ):
            sign_prior[c] = 1.0 if slope > 0.0 else -1.0
            count_score = min(1.0, float(c_idx.size) / float(max(min_samples * 2, 1)))
            corr_score = min(
                1.0, max(0.0, (abs(corr) - corr_thr) / max(1e-06, 1.0 - corr_thr))
            )
            slope_score = min(
                1.0, max(0.0, (abs(slope) - slope_thr) / max(1e-06, 4.0 * slope_thr))
            )
            span_score = min(
                1.0,
                max(0.0, (ph_span - span_thr) / max(1e-06, conf_span_ref - span_thr)),
            )
            conf_prior[c] = float(
                0.45 * corr_score
                + 0.3 * slope_score
                + 0.15 * count_score
                + 0.1 * span_score
            )
            valid_prior[c] = 1.0
    return {
        "sign": sign_prior.astype(np.float32, copy=False),
        "target_slope": target_slope_prior.astype(np.float32, copy=False),
        "confidence": conf_prior.astype(np.float32, copy=False),
        "ph_lo": ph_lo.astype(np.float32, copy=False),
        "ph_hi": ph_hi.astype(np.float32, copy=False),
        "slope": slope_prior.astype(np.float32, copy=False),
        "corr": corr_prior.astype(np.float32, copy=False),
        "valid": valid_prior.astype(np.float32, copy=False),
        "sample_sign": sample_sign.astype(np.float32, copy=False),
        "sample_target_slope": sample_target_slope.astype(np.float32, copy=False),
        "sample_confidence": sample_conf.astype(np.float32, copy=False),
        "sample_ph_lo": sample_ph_lo.astype(np.float32, copy=False),
        "sample_ph_hi": sample_ph_hi.astype(np.float32, copy=False),
        "sample_valid": sample_valid.astype(np.float32, copy=False),
    }


def _build_category_physics_prior(
    dataset: _data_dataset.FingerprintReactionDataset, train_idx: np.ndarray
) -> np.ndarray:
    return _build_category_physics_meta(dataset, train_idx)["sign"]


def _physics_meta_to_tensors(
    meta: Optional[Dict[str, np.ndarray]], device: str
) -> Dict[str, torch.Tensor]:
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, torch.Tensor] = {}
    for key, value in meta.items():
        arr = np.asarray(value, dtype=np.float32)
        out[key] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


def _group_mse_losses(
    pred: torch.Tensor, target: torch.Tensor, category_onehot: torch.Tensor
) -> torch.Tensor:
    cat_idx = torch.argmax(category_onehot, dim=1)
    uniq = torch.unique(cat_idx)
    losses = []
    for g in uniq:
        mask = cat_idx == g
        if torch.sum(mask).item() < 2:
            continue
        losses.append(F.mse_loss(pred[mask], target[mask], reduction="mean"))
    if not losses:
        return torch.empty((0,), device=pred.device, dtype=pred.dtype)
    return torch.stack(losses)


def compute_batch_losses(
    *,
    base_idx,
    category,
    dro_tau,
    fingerprint,
    lam_dro,
    lam_inv,
    lam_phys,
    numeric,
    phys_mono_w,
    phys_smooth_w,
    physics_colloc_w,
    physics_conf_min,
    physics_conf_power,
    physics_delta,
    physics_epoch_gate,
    physics_fd_conf_gated,
    physics_fd_range_gated,
    physics_fd_smooth_floor,
    physics_meta,
    physics_mode,
    physics_pinn_curv_conf_min,
    physics_pinn_curv_w,
    physics_safe_target,
    physics_sign_prior,
    physics_slope_clip,
    physics_slope_target_w,
    physics_use_colloc,
    trainer,
    use_amp,
    use_dro,
    use_invariant,
    use_physics,
    y,
):
    """Compute the unchanged prediction, invariance, DRO and physics losses for one batch."""
    components = None
    if (
        use_physics
        and physics_safe_target == "attn"
        and hasattr(trainer.model, "forward_components")
    ):
        components = trainer.model.forward_components(fingerprint, numeric)
        pred = components["final"]
        phys_pred_base = components["attn"]
    else:
        pred = trainer.model(fingerprint, numeric)
        phys_pred_base = pred
    pred_loss = F.mse_loss(pred, y, reduction="mean")
    group_losses = _group_mse_losses(pred, y, category)
    if use_invariant and group_losses.numel() > 1:
        inv_loss = torch.var(group_losses, unbiased=False)
    else:
        inv_loss = pred_loss.new_zeros(())
    if use_dro and group_losses.numel() > 0:
        tau = float(max(0.0001, dro_tau))
        dro_loss = float(tau) * torch.logsumexp(group_losses / float(tau), dim=0)
    else:
        dro_loss = pred_loss.new_zeros(())
    phys_loss = pred_loss.new_zeros(())
    if use_physics:
        if physics_mode == "finite_diff":
            numeric_pert = numeric.clone()
            numeric_pert[:, 0] = numeric_pert[:, 0] + float(physics_delta)
            if physics_safe_target == "attn" and hasattr(
                trainer.model, "forward_components"
            ):
                pert_components = trainer.model.forward_components(
                    fingerprint, numeric_pert
                )
                pred_plus = pert_components["attn"]
            else:
                pred_plus = trainer.model(fingerprint, numeric_pert)
            d_pred = pred_plus - phys_pred_base
            deriv_fd = d_pred / float(max(0.0001, abs(physics_delta)))
            smooth_loss = torch.mean(deriv_fd**2)
            mono_loss = pred_loss.new_zeros(())
            curv_weights = None
            curv_mask = None
            if physics_fd_conf_gated and physics_meta:
                cat_idx = torch.argmax(category, dim=1)
                sign = physics_meta.get("sign")
                conf = physics_meta.get("confidence")
                valid = physics_meta.get("valid")
                ph_lo = physics_meta.get("ph_lo")
                ph_hi = physics_meta.get("ph_hi")
                target_slope = physics_meta.get("target_slope")
                if sign is None:
                    sign = (
                        physics_sign_prior if physics_sign_prior is not None else None
                    )
                if sign is not None and sign.numel() > 0:
                    sign_b = sign[cat_idx].to(pred.dtype)
                    conf_b = (
                        conf[cat_idx].to(pred.dtype)
                        if conf is not None and conf.numel() > 0
                        else torch.ones_like(sign_b, dtype=pred.dtype)
                    )
                    valid_b = (
                        valid[cat_idx].to(torch.bool)
                        if valid is not None and valid.numel() > 0
                        else torch.abs(sign_b) > 0.5
                    )
                    target_b = (
                        target_slope[cat_idx].to(pred.dtype)
                        if target_slope is not None and target_slope.numel() > 0
                        else torch.zeros_like(sign_b, dtype=pred.dtype)
                    )
                    if physics_slope_clip > 0.0:
                        target_b = target_b.clamp(
                            -float(physics_slope_clip), float(physics_slope_clip)
                        )
                    if base_idx is not None:
                        sample_valid = physics_meta.get("sample_valid")
                        sample_sign = physics_meta.get("sample_sign")
                        sample_conf = physics_meta.get("sample_confidence")
                        sample_target = physics_meta.get("sample_target_slope")
                        sample_lo = physics_meta.get("sample_ph_lo")
                        sample_hi = physics_meta.get("sample_ph_hi")
                        if sample_valid is not None and sample_valid.numel() > 0:
                            safe_idx = base_idx.clamp(0, int(sample_valid.numel()) - 1)
                            sv = sample_valid[safe_idx].to(torch.bool)
                            if sample_sign is not None and sample_sign.numel() > 0:
                                sign_b = torch.where(
                                    sv, sample_sign[safe_idx].to(pred.dtype), sign_b
                                )
                            if sample_conf is not None and sample_conf.numel() > 0:
                                conf_b = torch.where(
                                    sv, sample_conf[safe_idx].to(pred.dtype), conf_b
                                )
                            if sample_target is not None and sample_target.numel() > 0:
                                target_b = torch.where(
                                    sv, sample_target[safe_idx].to(pred.dtype), target_b
                                )
                                if physics_slope_clip > 0.0:
                                    target_b = target_b.clamp(
                                        -float(physics_slope_clip),
                                        float(physics_slope_clip),
                                    )
                            valid_b = valid_b | sv
                    range_mask = torch.ones_like(valid_b, dtype=torch.bool)
                    if (
                        physics_fd_range_gated
                        and ph_lo is not None
                        and (ph_hi is not None)
                    ):
                        lo_b = ph_lo[cat_idx].to(pred.dtype)
                        hi_b = ph_hi[cat_idx].to(pred.dtype)
                        if base_idx is not None:
                            sample_valid = physics_meta.get("sample_valid")
                            sample_lo = physics_meta.get("sample_ph_lo")
                            sample_hi = physics_meta.get("sample_ph_hi")
                            if (
                                sample_valid is not None
                                and sample_valid.numel() > 0
                                and (sample_lo is not None)
                                and (sample_hi is not None)
                            ):
                                safe_idx = base_idx.clamp(
                                    0, int(sample_valid.numel()) - 1
                                )
                                sv = sample_valid[safe_idx].to(torch.bool)
                                lo_b = torch.where(
                                    sv, sample_lo[safe_idx].to(pred.dtype), lo_b
                                )
                                hi_b = torch.where(
                                    sv, sample_hi[safe_idx].to(pred.dtype), hi_b
                                )
                        ph_now = numeric[:, 0].to(pred.dtype)
                        ph_next = numeric_pert[:, 0].to(pred.dtype)
                        range_mask = (
                            (ph_now >= lo_b)
                            & (ph_now <= hi_b)
                            & (ph_next >= lo_b)
                            & (ph_next <= hi_b)
                        )
                        if float(physics_pinn_curv_w) > 1e-12:
                            ph_prev = ph_now - float(physics_delta)
                            range_mask = (
                                range_mask & (ph_prev >= lo_b) & (ph_prev <= hi_b)
                            )
                    smooth_floor = float(max(0.0, min(1.0, physics_fd_smooth_floor)))
                    conf_gate = conf_b.clamp(0.0, 1.0).pow(float(physics_conf_power))
                    smooth_weights = smooth_floor + (1.0 - smooth_floor) * conf_gate
                    if physics_fd_range_gated:
                        smooth_weights = smooth_weights * range_mask.to(pred.dtype)
                    curv_weights = smooth_weights
                    curv_mask = valid_b & (conf_b >= float(physics_pinn_curv_conf_min))
                    if physics_fd_range_gated:
                        curv_mask = curv_mask & range_mask
                    if torch.sum(smooth_weights).item() > 1e-08:
                        slope_mask = (
                            valid_b
                            & (torch.abs(sign_b) > 0.5)
                            & (conf_b >= float(physics_conf_min))
                        )
                        if physics_fd_range_gated:
                            slope_mask = slope_mask & range_mask
                        if (
                            torch.any(slope_mask)
                            and float(physics_slope_target_w) > 1e-12
                        ):
                            slope_weights = (
                                smooth_weights[slope_mask]
                                * conf_b[slope_mask].clamp_min(0.001)
                            ).to(pred.dtype)
                            target_loss = torch.sum(
                                slope_weights
                                * (deriv_fd[slope_mask] - target_b[slope_mask]) ** 2
                            ) / torch.sum(slope_weights)
                            floor_loss = torch.sum(
                                smooth_weights * deriv_fd**2
                            ) / torch.sum(smooth_weights)
                            stw = float(max(0.0, min(1.0, physics_slope_target_w)))
                            smooth_loss = stw * target_loss + (1.0 - stw) * floor_loss
                        else:
                            smooth_loss = torch.sum(
                                smooth_weights * deriv_fd**2
                            ) / torch.sum(smooth_weights)
                    mono_mask = (
                        (torch.abs(sign_b) > 0.5)
                        & valid_b
                        & range_mask
                        & (conf_b >= float(physics_conf_min))
                    )
                    if torch.any(mono_mask):
                        mono_weights = conf_b[mono_mask].clamp_min(0.001).to(pred.dtype)
                        mono_terms = torch.relu(
                            -(sign_b[mono_mask] * d_pred[mono_mask])
                        )
                        mono_loss = torch.sum(mono_weights * mono_terms) / torch.sum(
                            mono_weights
                        )
            elif physics_sign_prior is not None and physics_sign_prior.numel() > 0:
                cat_idx = torch.argmax(category, dim=1)
                sign = physics_sign_prior[cat_idx].to(pred.dtype)
                mask = torch.abs(sign) > 0.5
                if torch.any(mask):
                    mono_loss = torch.mean(torch.relu(-(sign[mask] * d_pred[mask])))
            phys_loss = (
                float(phys_smooth_w) * smooth_loss + float(phys_mono_w) * mono_loss
            )
            if float(physics_pinn_curv_w) > 1e-12:
                numeric_minus = numeric.clone()
                numeric_minus[:, 0] = numeric_minus[:, 0] - float(physics_delta)
                if physics_safe_target == "attn" and hasattr(
                    trainer.model, "forward_components"
                ):
                    minus_components = trainer.model.forward_components(
                        fingerprint, numeric_minus
                    )
                    pred_minus = minus_components["attn"]
                else:
                    pred_minus = trainer.model(fingerprint, numeric_minus)
                denom = float(max(0.0001, abs(physics_delta)) ** 2)
                d2_fd = (pred_plus - 2.0 * phys_pred_base + pred_minus) / denom
                if curv_weights is None:
                    curv_weights = torch.ones_like(d2_fd, dtype=pred.dtype)
                if curv_mask is None:
                    curv_mask = torch.ones_like(d2_fd, dtype=torch.bool)
                curv_mask = curv_mask.reshape_as(d2_fd).to(torch.bool)
                curv_weights = curv_weights.reshape_as(d2_fd).to(pred.dtype)
                if torch.any(curv_mask):
                    cw = curv_weights[curv_mask].clamp_min(0.001)
                    curv_loss = torch.sum(cw * d2_fd[curv_mask] ** 2) / torch.sum(cw)
                    phys_loss = (
                        phys_loss + float(max(0.0, physics_pinn_curv_w)) * curv_loss
                    )
        else:
            physics_ctx = (
                torch.cuda.amp.autocast(enabled=False) if use_amp else nullcontext()
            )
            with physics_ctx:
                fp_phys = fingerprint.to(torch.float32)
                num_base = numeric.to(torch.float32)
                ph = num_base[:, :1].detach().clone().requires_grad_(True)
                cat_feat = num_base[:, 1:].detach()
                numeric_pi = torch.cat([ph, cat_feat], dim=1)
                pred_pi = trainer.model(fp_phys, numeric_pi).to(torch.float32)
                dy_dph = torch.autograd.grad(pred_pi.sum(), ph, create_graph=True)[
                    0
                ].reshape(-1)
                cat_idx = torch.argmax(category, dim=1)
                sign = physics_meta.get("sign")
                conf = physics_meta.get("confidence")
                valid = physics_meta.get("valid")
                target_slope = physics_meta.get("target_slope")
                if sign is None:
                    sign = torch.zeros((1,), dtype=torch.float32, device=pred.device)
                if conf is None:
                    conf = torch.ones_like(sign)
                if valid is None:
                    valid = (torch.abs(sign) > 0.5).to(torch.float32)
                sign_b = sign[cat_idx].to(torch.float32)
                conf_b = conf[cat_idx].to(torch.float32)
                valid_b = valid[cat_idx].to(torch.float32)
                target_b = (
                    target_slope[cat_idx].to(torch.float32)
                    if target_slope is not None and target_slope.numel() > 0
                    else torch.zeros_like(sign_b, dtype=torch.float32)
                )
                if physics_slope_clip > 0.0:
                    target_b = target_b.clamp(
                        -float(physics_slope_clip), float(physics_slope_clip)
                    )
                if base_idx is not None:
                    sample_valid = physics_meta.get("sample_valid")
                    sample_sign = physics_meta.get("sample_sign")
                    sample_conf = physics_meta.get("sample_confidence")
                    sample_target = physics_meta.get("sample_target_slope")
                    if sample_valid is not None and sample_valid.numel() > 0:
                        safe_idx = base_idx.clamp(0, int(sample_valid.numel()) - 1)
                        sv = sample_valid[safe_idx].to(torch.bool)
                        if sample_sign is not None and sample_sign.numel() > 0:
                            sign_b = torch.where(
                                sv, sample_sign[safe_idx].to(torch.float32), sign_b
                            )
                        if sample_conf is not None and sample_conf.numel() > 0:
                            conf_b = torch.where(
                                sv, sample_conf[safe_idx].to(torch.float32), conf_b
                            )
                        if sample_target is not None and sample_target.numel() > 0:
                            target_b = torch.where(
                                sv, sample_target[safe_idx].to(torch.float32), target_b
                            )
                            if physics_slope_clip > 0.0:
                                target_b = target_b.clamp(
                                    -float(physics_slope_clip),
                                    float(physics_slope_clip),
                                )
                        valid_b = torch.maximum(valid_b, sv.to(torch.float32))
                mask = (
                    (torch.abs(sign_b) > 0.5)
                    & (conf_b >= float(physics_conf_min))
                    & (valid_b > 0.5)
                )
                smooth_loss = torch.mean(dy_dph**2)
                mono_loss = pred_pi.new_zeros(())
                if torch.any(mask):
                    weights = conf_b[mask].clamp_min(0.001)
                    mono_terms = torch.relu(-(sign_b[mask] * dy_dph[mask]))
                    mono_loss = torch.sum(weights * mono_terms) / torch.sum(weights)
                    target_loss = torch.sum(
                        weights * (dy_dph[mask] - target_b[mask]) ** 2
                    ) / torch.sum(weights)
                    floor_loss = torch.sum(weights * dy_dph[mask] ** 2) / torch.sum(
                        weights
                    )
                    stw = float(max(0.0, min(1.0, physics_slope_target_w)))
                    smooth_loss = stw * target_loss + (1.0 - stw) * floor_loss
                phys_loss = (
                    float(phys_smooth_w) * smooth_loss + float(phys_mono_w) * mono_loss
                )
                if physics_use_colloc and torch.any(mask):
                    ph_lo = physics_meta.get("ph_lo")
                    ph_hi = physics_meta.get("ph_hi")
                    if ph_lo is not None and ph_hi is not None:
                        lo_b = ph_lo[cat_idx].to(torch.float32).unsqueeze(-1)
                        hi_b = ph_hi[cat_idx].to(torch.float32).unsqueeze(-1)
                        span = (hi_b - lo_b).clamp_min(0.001)
                        rand = torch.rand_like(ph)
                        ph_col = lo_b + rand * span
                        ph_col = torch.where(mask.unsqueeze(-1), ph_col, ph.detach())
                        ph_col = ph_col.detach().clone().requires_grad_(True)
                        numeric_col = torch.cat([ph_col, cat_feat], dim=1)
                        pred_col = trainer.model(fp_phys, numeric_col).to(torch.float32)
                        dy_col = torch.autograd.grad(
                            pred_col.sum(), ph_col, create_graph=True
                        )[0].reshape(-1)
                        weights = conf_b[mask].clamp_min(0.001)
                        colloc_target = torch.sum(
                            weights * (dy_col[mask] - target_b[mask]) ** 2
                        ) / torch.sum(weights)
                        colloc_floor = torch.sum(
                            weights * dy_col[mask] ** 2
                        ) / torch.sum(weights)
                        stw = float(max(0.0, min(1.0, physics_slope_target_w)))
                        colloc_smooth = stw * colloc_target + (1.0 - stw) * colloc_floor
                        colloc_mono = torch.sum(
                            weights * torch.relu(-(sign_b[mask] * dy_col[mask]))
                        ) / torch.sum(weights)
                        colloc_loss = (
                            float(phys_smooth_w) * colloc_smooth
                            + float(phys_mono_w) * colloc_mono
                        )
                        phys_loss = (
                            phys_loss + float(max(0.0, physics_colloc_w)) * colloc_loss
                        )
    effective_lam_phys = float(lam_phys) * float(physics_epoch_gate)
    total_loss = (
        pred_loss
        + float(lam_inv) * inv_loss
        + float(lam_dro) * dro_loss
        + effective_lam_phys * phys_loss
    )
    return (total_loss, pred_loss, inv_loss, dro_loss, phys_loss)
