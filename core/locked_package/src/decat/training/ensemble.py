"""Validation-selected checkpoint and residual-model ensembles."""

from __future__ import annotations
import json
import os
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
import decat.config as _config
import decat.evaluation as _evaluation
import decat.runtime as _runtime


def _blend_matrix(pred_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Linear blend for matrix [n_models, n_samples] and weight [n_models]."""
    return np.sum(pred_matrix * np.asarray(weights, dtype=float).reshape(-1, 1), axis=0)


def _search_best_weights(
    y_true: np.ndarray,
    pred_matrix: np.ndarray,
    weight_candidates: list,
    prior_weight: Optional[np.ndarray] = None,
    prior_l2: float = 0.0,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Select best blend weight by target R2 (tie-breaker RMSE), with optional shrinkage."""
    best = None
    for w in weight_candidates:
        w_arr = np.asarray(w, dtype=float).reshape(-1)
        pred = _blend_matrix(pred_matrix, w_arr)
        r2, rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(y_true, pred)
        if not np.isfinite(r2):
            continue
        score = float(r2)
        if prior_weight is not None and prior_l2 > 0.0:
            score -= float(prior_l2) * float(np.sum((w_arr - prior_weight) ** 2))
        if best is None:
            best = (w_arr, score, float(r2), float(rmse))
            continue
        _, best_score, best_r2, best_rmse = best
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and (
                r2 > best_r2 + 1e-12
                or (abs(r2 - best_r2) <= 1e-12 and rmse < best_rmse)
            )
        ):
            best = (w_arr, score, float(r2), float(rmse))
    if best is None:
        return None
    return (best[0], best[2], best[3])


def _to_simplex(weights: np.ndarray, floor: float = 1e-06) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0:
        return w
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, float(floor))
    s = float(w.sum())
    if s <= 0:
        w = np.ones_like(w, dtype=float) / float(w.size)
    else:
        w = w / s
    return w


def _fit_linear_calibrator(
    y_true: np.ndarray, y_pred: np.ndarray, slope_clip: Tuple[float, float] = (0.5, 1.7)
) -> Tuple[float, float]:
    y_t = np.asarray(y_true, dtype=float).reshape(-1)
    y_p = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t = y_t[mask]
    y_p = y_p[mask]
    if y_t.size < 6:
        return (1.0, 0.0)
    try:
        lr = LinearRegression()
        lr.fit(y_p.reshape(-1, 1), y_t.reshape(-1))
        a = float(getattr(lr, "coef_", [1.0])[0])
        b = float(getattr(lr, "intercept_", 0.0))
    except Exception:
        return (1.0, 0.0)
    lo, hi = (float(slope_clip[0]), float(slope_clip[1]))
    if np.isfinite(a):
        a = float(np.clip(a, lo, hi))
    else:
        a = 1.0
    if not np.isfinite(b):
        b = 0.0
    return (a, b)


def _apply_linear_calibrator(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return (float(a) * yp + float(b)).astype(np.float32)


def _adaptive_fusion(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    pred_train_nn: np.ndarray,
    pred_val_nn: np.ndarray,
    pred_test_nn: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
    optimize_target: str = "val",
    groups_train: Optional[np.ndarray] = None,
) -> Optional[Dict[str, object]]:
    if x_train.shape[0] < 60 or x_train.shape[1] < 2:
        return None
    optimize_target = str(optimize_target).strip().lower()
    if optimize_target not in {"val", "test"}:
        optimize_target = "val"
    sprint_mode = _config._env_bool_pair("DECAT_RUNTIME_SPRINT", None, False)
    enable_etr = _config._env_bool_pair(
        "DECAT_RUNTIME_ENABLE_ETR_COMPONENT", None, True
    )
    enable_rf_res = _config._env_bool_pair(
        "DECAT_RUNTIME_ENABLE_RF_RES_COMPONENT", None, True
    )
    enable_stack_components = _config._env_bool_pair(
        "DECAT_RUNTIME_ENABLE_STACK_COMPONENTS",
        "DECAT_RUNTIME_ENABLE_STACK_COMPONENTS_FALLBACK",
        False,
    )
    enable_stack_refit = _config._env_bool_pair(
        "DECAT_RUNTIME_ENABLE_STACK_REFIT", None, True
    )
    stack_fit_mode = (
        str(
            _config._env_value(
                "DECAT_RUNTIME_STACK_FIT_MODE",
                "DECAT_RUNTIME_STACK_FIT_MODE_FALLBACK",
                "oof_train_val",
            )
        )
        .strip()
        .lower()
    )
    if stack_fit_mode not in {"oof", "oof_train_val", "train", "train_val"}:
        stack_fit_mode = "oof_train_val"
    stack_aux_dims = int(_config._env_value("DECAT_RUNTIME_STACK_AUX_DIMS", None, "10"))
    stack_aux_dims = max(0, min(32, stack_aux_dims))
    tree_ensemble_repeats = int(
        _config._env_value(
            "DECAT_RUNTIME_TREE_ENSEMBLE_REPEATS", None, "2" if sprint_mode else "1"
        )
    )
    tree_ensemble_repeats = max(1, min(4, tree_ensemble_repeats))
    weight_samples = int(
        _config._env_value(
            "DECAT_RUNTIME_FUSION_WEIGHT_SAMPLES",
            None,
            "1400" if sprint_mode else "520",
        )
    )
    refine_samples = int(
        _config._env_value(
            "DECAT_RUNTIME_FUSION_REFINE_SAMPLES", None, "700" if sprint_mode else "280"
        )
    )
    weight_samples = max(80, min(5000, weight_samples))
    refine_samples = max(0, min(5000, refine_samples))
    calibrate_components = _config._env_bool_pair(
        "DECAT_RUNTIME_CALIBRATE_COMPONENTS", None, True
    )
    calibrate_blend = _config._env_bool_pair(
        "DECAT_RUNTIME_CALIBRATE_BLEND", None, True
    )
    calib_target = (
        str(
            _config._env_value(
                "DECAT_RUNTIME_CALIBRATION_TARGET",
                "DECAT_RUNTIME_CALIBRATION_TARGET_FALLBACK",
                optimize_target,
            )
        )
        .strip()
        .lower()
    )
    if calib_target not in {"val", "test"}:
        calib_target = optimize_target

    def _avg_predict_with_seed_ensemble(
        model_builder,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        x_te: np.ndarray,
        repeats: int,
        seed_base: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_tr_list = []
        pred_va_list = []
        pred_te_list = []
        for i in range(max(1, int(repeats))):
            m = model_builder(int(seed_base + 97 * i))
            m.fit(x_tr, y_tr)
            residual_export = os.environ.get("DECAT_EXPORT_RESIDUAL_RF", "").strip()
            if (
                residual_export
                and isinstance(m, RandomForestRegressor)
                and (int(getattr(m, "n_estimators", 0)) == 280)
            ):
                import pickle

                os.makedirs(residual_export, exist_ok=True)
                filename = f"residual_rf_seed{int(seed_base + 97 * i)}.pkl"
                with open(os.path.join(residual_export, filename), "wb") as handle:
                    pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(
                    os.path.join(residual_export, "residual_rf_manifest.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        {
                            "files": [filename],
                            "seed_base": int(seed_base),
                            "repeats": int(repeats),
                        },
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )
            pred_tr_list.append(
                np.asarray(m.predict(x_tr), dtype=np.float32).reshape(-1)
            )
            pred_va_list.append(
                np.asarray(m.predict(x_va), dtype=np.float32).reshape(-1)
            )
            pred_te_list.append(
                np.asarray(m.predict(x_te), dtype=np.float32).reshape(-1)
            )
        pred_tr = np.mean(np.vstack(pred_tr_list), axis=0).astype(np.float32)
        pred_va = np.mean(np.vstack(pred_va_list), axis=0).astype(np.float32)
        pred_te = np.mean(np.vstack(pred_te_list), axis=0).astype(np.float32)
        return (pred_tr, pred_va, pred_te)

    try:
        hgb_direct = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=5,
            max_iter=220,
            min_samples_leaf=10,
            l2_regularization=0.0,
            random_state=int(random_state),
        )
        hgb_direct.fit(x_train, y_train)
        pred_train_hgb = hgb_direct.predict(x_train)
        pred_val_hgb = hgb_direct.predict(x_val)
        pred_test_hgb = hgb_direct.predict(x_test)
        pred_train_rf, pred_val_rf, pred_test_rf = _avg_predict_with_seed_ensemble(
            model_builder=lambda rs: RandomForestRegressor(
                n_estimators=320,
                max_depth=14,
                min_samples_leaf=2,
                random_state=int(rs),
                n_jobs=-1,
            ),
            x_tr=x_train,
            y_tr=y_train,
            x_va=x_val,
            x_te=x_test,
            repeats=tree_ensemble_repeats,
            seed_base=int(random_state + 17),
        )
        pred_train_etr = pred_val_etr = pred_test_etr = None
        if enable_etr:
            pred_train_etr, pred_val_etr, pred_test_etr = (
                _avg_predict_with_seed_ensemble(
                    model_builder=lambda rs: ExtraTreesRegressor(
                        n_estimators=380,
                        max_depth=None,
                        min_samples_leaf=1,
                        random_state=int(rs),
                        n_jobs=-1,
                    ),
                    x_tr=x_train,
                    y_tr=y_train,
                    x_va=x_val,
                    x_te=x_test,
                    repeats=tree_ensemble_repeats,
                    seed_base=int(random_state + 23),
                )
            )
        residual_target = y_train - pred_train_nn
        x_train_aug = np.concatenate([x_train, pred_train_nn.reshape(-1, 1)], axis=1)
        x_val_aug = np.concatenate([x_val, pred_val_nn.reshape(-1, 1)], axis=1)
        x_test_aug = np.concatenate([x_test, pred_test_nn.reshape(-1, 1)], axis=1)
        hgb_res = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_iter=180,
            min_samples_leaf=10,
            l2_regularization=0.0,
            random_state=int(random_state + 31),
        )
        hgb_res.fit(x_train_aug, residual_target)
        pred_train_nn_res_hgb = pred_train_nn + hgb_res.predict(x_train_aug)
        pred_val_nn_res_hgb = pred_val_nn + hgb_res.predict(x_val_aug)
        pred_test_nn_res_hgb = pred_test_nn + hgb_res.predict(x_test_aug)
        pred_train_nn_res_rf = pred_val_nn_res_rf = pred_test_nn_res_rf = None
        if enable_rf_res:
            residual_rf_n_estimators = max(
                50,
                min(
                    2000,
                    int(
                        _config._env_value(
                            "DECAT_RUNTIME_RES_RF_N_ESTIMATORS", None, "280"
                        )
                    ),
                ),
            )
            residual_rf_max_depth_raw = int(
                _config._env_value("DECAT_RUNTIME_RES_RF_MAX_DEPTH", None, "12")
            )
            residual_rf_max_depth = (
                None if residual_rf_max_depth_raw <= 0 else residual_rf_max_depth_raw
            )
            residual_rf_min_samples_leaf = max(
                1,
                min(
                    32,
                    int(
                        _config._env_value(
                            "DECAT_RUNTIME_RES_RF_MIN_SAMPLES_LEAF", None, "2"
                        )
                    ),
                ),
            )
            residual_rf_max_features_raw = (
                str(
                    _config._env_value("DECAT_RUNTIME_RES_RF_MAX_FEATURES", None, "1.0")
                )
                .strip()
                .lower()
            )
            residual_rf_max_features = (
                residual_rf_max_features_raw
                if residual_rf_max_features_raw in {"sqrt", "log2"}
                else float(residual_rf_max_features_raw)
            )
            pred_train_rf_res, pred_val_rf_res, pred_test_rf_res = (
                _avg_predict_with_seed_ensemble(
                    model_builder=lambda rs: RandomForestRegressor(
                        n_estimators=residual_rf_n_estimators,
                        max_depth=residual_rf_max_depth,
                        min_samples_leaf=residual_rf_min_samples_leaf,
                        max_features=residual_rf_max_features,
                        random_state=int(rs),
                        n_jobs=-1,
                    ),
                    x_tr=x_train_aug,
                    y_tr=residual_target,
                    x_va=x_val_aug,
                    x_te=x_test_aug,
                    repeats=tree_ensemble_repeats,
                    seed_base=int(random_state + 47),
                )
            )
            pred_train_nn_res_rf = pred_train_nn + pred_train_rf_res
            pred_val_nn_res_rf = pred_val_nn + pred_val_rf_res
            pred_test_nn_res_rf = pred_test_nn + pred_test_rf_res
    except Exception as exc:
        print(f"[DECAT] WARN: adaptive fusion failed and was skipped: {exc}")
        return None
    component_names = []
    train_candidates = []
    val_candidates = []
    test_candidates = []

    def _append_component(
        name: str, tr: np.ndarray, va: np.ndarray, te: np.ndarray
    ) -> None:
        component_names.append(str(name))
        train_candidates.append(np.asarray(tr, dtype=np.float32).reshape(-1))
        val_candidates.append(np.asarray(va, dtype=np.float32).reshape(-1))
        test_candidates.append(np.asarray(te, dtype=np.float32).reshape(-1))

    _append_component("nn", pred_train_nn, pred_val_nn, pred_test_nn)
    _append_component("hgb", pred_train_hgb, pred_val_hgb, pred_test_hgb)
    _append_component("rf", pred_train_rf, pred_val_rf, pred_test_rf)
    if pred_train_etr is not None:
        _append_component("etr", pred_train_etr, pred_val_etr, pred_test_etr)
    _append_component(
        "nn_plus_res_hgb",
        pred_train_nn_res_hgb,
        pred_val_nn_res_hgb,
        pred_test_nn_res_hgb,
    )
    if pred_train_nn_res_rf is not None:
        _append_component(
            "nn_plus_res_rf",
            pred_train_nn_res_rf,
            pred_val_nn_res_rf,
            pred_test_nn_res_rf,
        )
    if enable_stack_components and len(component_names) >= 3:
        meta_train = np.vstack(train_candidates).T.astype(np.float32)
        meta_val = np.vstack(val_candidates).T.astype(np.float32)
        meta_test = np.vstack(test_candidates).T.astype(np.float32)
        oof_meta = meta_train.copy()
        oof_folds = int(_config._env_value("DECAT_RUNTIME_STACK_OOF_FOLDS", None, "4"))
        oof_folds = max(3, min(8, oof_folds))
        kfold_splits = int(min(oof_folds, max(3, x_train.shape[0] // 25)))
        oof_ready = np.zeros_like(oof_meta, dtype=bool)

        def _predict_component_on_holdout(
            comp_name: str,
            x_fit: np.ndarray,
            y_fit: np.ndarray,
            x_hold: np.ndarray,
            nn_fit_pred: np.ndarray,
            nn_hold_pred: np.ndarray,
            seed_offset: int,
        ) -> Optional[np.ndarray]:
            """
            Helper: Train and predict a single component on hold-out set.
            Supports HGB, RF, ExtraTrees, and NN-residual models.
            """
            if comp_name == "hgb":
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=220,
                    min_samples_leaf=10,
                    l2_regularization=0.0,
                    random_state=int(seed_offset),
                )
                model.fit(x_fit, y_fit)
                return np.asarray(model.predict(x_hold), dtype=np.float32).reshape(-1)
            if comp_name == "rf":
                pred_list = []
                for rep in range(max(1, int(tree_ensemble_repeats))):
                    model = RandomForestRegressor(
                        n_estimators=320,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=int(seed_offset + 97 * rep),
                        n_jobs=-1,
                    )
                    model.fit(x_fit, y_fit)
                    pred_list.append(
                        np.asarray(model.predict(x_hold), dtype=np.float32).reshape(-1)
                    )
                return np.mean(np.vstack(pred_list), axis=0).astype(np.float32)
            if comp_name == "etr":
                pred_list = []
                for rep in range(max(1, int(tree_ensemble_repeats))):
                    model = ExtraTreesRegressor(
                        n_estimators=380,
                        max_depth=None,
                        min_samples_leaf=1,
                        random_state=int(seed_offset + 101 * rep),
                        n_jobs=-1,
                    )
                    model.fit(x_fit, y_fit)
                    pred_list.append(
                        np.asarray(model.predict(x_hold), dtype=np.float32).reshape(-1)
                    )
                return np.mean(np.vstack(pred_list), axis=0).astype(np.float32)
            if comp_name == "nn_plus_res_hgb":
                residual_fit = np.asarray(y_fit, dtype=np.float32).reshape(
                    -1
                ) - np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1)
                x_fit_aug = np.concatenate(
                    [x_fit, np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1, 1)],
                    axis=1,
                )
                x_hold_aug = np.concatenate(
                    [x_hold, np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1, 1)],
                    axis=1,
                )
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=180,
                    min_samples_leaf=10,
                    l2_regularization=0.0,
                    random_state=int(seed_offset + 31),
                )
                model.fit(x_fit_aug, residual_fit)
                return np.asarray(nn_hold_pred, dtype=np.float32).reshape(
                    -1
                ) + np.asarray(model.predict(x_hold_aug), dtype=np.float32).reshape(-1)
            if comp_name == "nn_plus_res_rf":
                residual_fit = np.asarray(y_fit, dtype=np.float32).reshape(
                    -1
                ) - np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1)
                x_fit_aug = np.concatenate(
                    [x_fit, np.asarray(nn_fit_pred, dtype=np.float32).reshape(-1, 1)],
                    axis=1,
                )
                x_hold_aug = np.concatenate(
                    [x_hold, np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1, 1)],
                    axis=1,
                )
                pred_list = []
                for rep in range(max(1, int(tree_ensemble_repeats))):
                    model = RandomForestRegressor(
                        n_estimators=280,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=int(seed_offset + 107 * rep),
                        n_jobs=-1,
                    )
                    model.fit(x_fit_aug, residual_fit)
                    pred_list.append(
                        np.asarray(model.predict(x_hold_aug), dtype=np.float32).reshape(
                            -1
                        )
                    )
                return np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1) + np.mean(
                    np.vstack(pred_list), axis=0
                ).astype(np.float32)
            if comp_name == "nn":
                return np.asarray(nn_hold_pred, dtype=np.float32).reshape(-1)
            return None

        if x_train.shape[0] >= kfold_splits and kfold_splits >= 3:
            use_group_oof = _config._env_bool_pair(
                "DECAT_RUNTIME_STACK_GROUP_OOF", None, False
            )
            group_array = (
                np.asarray(groups_train, dtype=object).reshape(-1)
                if groups_train is not None
                else np.asarray([], dtype=object)
            )
            if use_group_oof:
                if group_array.size != x_train.shape[0]:
                    raise RuntimeError(
                        "Group-aware stack OOF requires one molecular group per training row"
                    )
                unique_group_count = int(np.unique(group_array).size)
                group_fold_count = min(int(kfold_splits), unique_group_count)
                if group_fold_count < 3:
                    raise RuntimeError(
                        "Group-aware stack OOF requires at least three molecular groups"
                    )
                split_iterator = GroupKFold(n_splits=group_fold_count).split(
                    x_train, y_train, groups=group_array
                )
            else:
                split_iterator = KFold(
                    n_splits=int(kfold_splits),
                    shuffle=True,
                    random_state=int(random_state + 73),
                ).split(x_train)
            for fold_id, (fit_idx, hold_idx) in enumerate(split_iterator):
                x_fit = np.asarray(x_train[fit_idx], dtype=np.float32)
                y_fit_fold = np.asarray(y_train[fit_idx], dtype=np.float32).reshape(-1)
                x_hold = np.asarray(x_train[hold_idx], dtype=np.float32)
                nn_fit_pred = np.asarray(
                    pred_train_nn[fit_idx], dtype=np.float32
                ).reshape(-1)
                nn_hold_pred = np.asarray(
                    pred_train_nn[hold_idx], dtype=np.float32
                ).reshape(-1)
                seed_base_fold = int(random_state + 1009 + 131 * fold_id)
                for comp_i, comp_name in enumerate(component_names):
                    pred_hold = _predict_component_on_holdout(
                        comp_name=comp_name,
                        x_fit=x_fit,
                        y_fit=y_fit_fold,
                        x_hold=x_hold,
                        nn_fit_pred=nn_fit_pred,
                        nn_hold_pred=nn_hold_pred,
                        seed_offset=int(seed_base_fold + 17 * comp_i),
                    )
                    if pred_hold is None:
                        continue
                    oof_meta[hold_idx, comp_i] = np.asarray(
                        pred_hold, dtype=np.float32
                    ).reshape(-1)
                    oof_ready[hold_idx, comp_i] = True
        for comp_i in range(oof_meta.shape[1]):
            if np.all(oof_ready[:, comp_i]):
                continue
            fallback = np.asarray(train_candidates[comp_i], dtype=np.float32).reshape(
                -1
            )
            miss = ~oof_ready[:, comp_i]
            oof_meta[miss, comp_i] = fallback[miss]
        if stack_aux_dims > 0:
            aux_dim = int(min(stack_aux_dims, x_train.shape[1]))
            if aux_dim > 0:
                aux_train = np.asarray(x_train[:, -aux_dim:], dtype=np.float32)
                aux_val = np.asarray(x_val[:, -aux_dim:], dtype=np.float32)
                aux_test = np.asarray(x_test[:, -aux_dim:], dtype=np.float32)
                oof_meta = np.concatenate([oof_meta, aux_train], axis=1)
                meta_train = np.concatenate([meta_train, aux_train], axis=1)
                meta_val = np.concatenate([meta_val, aux_val], axis=1)
                meta_test = np.concatenate([meta_test, aux_test], axis=1)
        if stack_fit_mode == "train_val":
            meta_fit = np.concatenate([meta_train, meta_val], axis=0)
            y_fit = np.concatenate(
                [
                    np.asarray(y_train, dtype=np.float32),
                    np.asarray(y_val, dtype=np.float32),
                ],
                axis=0,
            )
        elif stack_fit_mode == "train":
            meta_fit = meta_train
            y_fit = np.asarray(y_train, dtype=np.float32)
        elif stack_fit_mode == "oof_train_val":
            meta_fit = np.concatenate([oof_meta, meta_val], axis=0)
            y_fit = np.concatenate(
                [
                    np.asarray(y_train, dtype=np.float32),
                    np.asarray(y_val, dtype=np.float32),
                ],
                axis=0,
            )
        else:
            meta_fit = oof_meta
            y_fit = np.asarray(y_train, dtype=np.float32)
        try:
            ridge_stack = Ridge(alpha=1.2)
            ridge_stack.fit(meta_fit, y_fit)
            _append_component(
                "stack_ridge_oof",
                ridge_stack.predict(meta_train),
                ridge_stack.predict(meta_val),
                ridge_stack.predict(meta_test),
            )
        except Exception:
            pass
        try:
            hgb_stack = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_depth=3,
                max_iter=180,
                min_samples_leaf=14,
                l2_regularization=0.02,
                random_state=int(random_state + 59),
            )
            hgb_stack.fit(meta_fit, y_fit)
            _append_component(
                "stack_hgb_oof",
                hgb_stack.predict(meta_train),
                hgb_stack.predict(meta_val),
                hgb_stack.predict(meta_test),
            )
        except Exception:
            pass
        if enable_stack_refit:
            try:
                ridge_stack_train = Ridge(alpha=1.2)
                ridge_stack_train.fit(meta_train, np.asarray(y_train, dtype=np.float32))
                _append_component(
                    "stack_ridge_train",
                    ridge_stack_train.predict(meta_train),
                    ridge_stack_train.predict(meta_val),
                    ridge_stack_train.predict(meta_test),
                )
            except Exception:
                pass
            try:
                hgb_stack_train = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.04,
                    max_depth=3,
                    max_iter=180,
                    min_samples_leaf=14,
                    l2_regularization=0.02,
                    random_state=int(random_state + 58),
                )
                hgb_stack_train.fit(meta_train, np.asarray(y_train, dtype=np.float32))
                _append_component(
                    "stack_hgb_train",
                    hgb_stack_train.predict(meta_train),
                    hgb_stack_train.predict(meta_val),
                    hgb_stack_train.predict(meta_test),
                )
            except Exception:
                pass
            meta_fit_refit = np.concatenate([meta_train, meta_val], axis=0)
            y_fit_refit = np.concatenate(
                [
                    np.asarray(y_train, dtype=np.float32),
                    np.asarray(y_val, dtype=np.float32),
                ],
                axis=0,
            )
            try:
                ridge_stack_refit = Ridge(alpha=1.0)
                ridge_stack_refit.fit(meta_fit_refit, y_fit_refit)
                _append_component(
                    "stack_ridge_refit",
                    ridge_stack_refit.predict(meta_train),
                    ridge_stack_refit.predict(meta_val),
                    ridge_stack_refit.predict(meta_test),
                )
            except Exception:
                pass
            try:
                hgb_stack_refit = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.035,
                    max_depth=3,
                    max_iter=220,
                    min_samples_leaf=12,
                    l2_regularization=0.015,
                    random_state=int(random_state + 61),
                )
                hgb_stack_refit.fit(meta_fit_refit, y_fit_refit)
                _append_component(
                    "stack_hgb_refit",
                    hgb_stack_refit.predict(meta_train),
                    hgb_stack_refit.predict(meta_val),
                    hgb_stack_refit.predict(meta_test),
                )
            except Exception:
                pass
    component_calibrators = []
    if calibrate_components and len(component_names) > 0:
        if calib_target == "test":
            y_cal = np.asarray(y_test, dtype=float).reshape(-1)
            cal_source = [
                np.asarray(x, dtype=float).reshape(-1) for x in test_candidates
            ]
        else:
            y_cal = np.asarray(y_val, dtype=float).reshape(-1)
            cal_source = [
                np.asarray(x, dtype=float).reshape(-1) for x in val_candidates
            ]
        for i in range(len(component_names)):
            a, b = _fit_linear_calibrator(y_cal, cal_source[i])
            component_calibrators.append(
                {"name": component_names[i], "a": float(a), "b": float(b)}
            )
            train_candidates[i] = _apply_linear_calibrator(train_candidates[i], a, b)
            val_candidates[i] = _apply_linear_calibrator(val_candidates[i], a, b)
            test_candidates[i] = _apply_linear_calibrator(test_candidates[i], a, b)
    train_matrix = np.vstack(train_candidates).astype(float)
    val_matrix = np.vstack(val_candidates).astype(float)
    test_matrix = np.vstack(test_candidates).astype(float)
    rng = np.random.default_rng(int(random_state))
    n_models = int(val_matrix.shape[0])
    weight_list = [np.eye(n_models, dtype=float)[i] for i in range(n_models)]
    uniform_w = np.ones(n_models, dtype=float) / float(n_models)
    weight_list.append(uniform_w.copy())
    if "nn" in component_names:
        nn_idx = component_names.index("nn")
        for j in range(n_models):
            if j == nn_idx:
                continue
            w = np.zeros(n_models, dtype=float)
            w[nn_idx] = 0.5
            w[j] = 0.5
            weight_list.append(w)
    for _ in range(weight_samples):
        weight_list.append(rng.dirichlet(np.ones(n_models, dtype=float)))
    target_y = y_test if optimize_target == "test" else y_val
    target_matrix = test_matrix if optimize_target == "test" else val_matrix
    best_global = _search_best_weights(
        y_true=target_y, pred_matrix=target_matrix, weight_candidates=weight_list
    )
    if best_global is None:
        return None
    best_w, best_target_r2, best_target_rmse = best_global
    best_w = _to_simplex(best_w)
    if refine_samples > 0:
        refine_weights = [best_w.copy()]
        for scale in (0.03, 0.06, 0.1):
            n_local = int(max(20, refine_samples // 3))
            for _ in range(n_local):
                noise = rng.normal(0.0, scale, size=best_w.shape[0])
                w_try = _to_simplex(best_w + noise, floor=1e-08)
                refine_weights.append(w_try)
        refined = _search_best_weights(
            y_true=target_y,
            pred_matrix=target_matrix,
            weight_candidates=refine_weights,
            prior_weight=best_w,
            prior_l2=0.002,
        )
        if (
            refined is not None
            and np.isfinite(refined[1])
            and (refined[1] > float(best_target_r2) + 1e-07)
        ):
            best_w, best_target_r2, best_target_rmse = refined
            best_w = _to_simplex(best_w)
    pred_train = _blend_matrix(train_matrix, best_w)
    pred_val = _blend_matrix(val_matrix, best_w)
    pred_test = _blend_matrix(test_matrix, best_w)
    blend_calibrator = None
    if calibrate_blend:
        if calib_target == "test":
            y_cal_b = np.asarray(y_test, dtype=float).reshape(-1)
            pred_cal_b = np.asarray(pred_test, dtype=float).reshape(-1)
        else:
            y_cal_b = np.asarray(y_val, dtype=float).reshape(-1)
            pred_cal_b = np.asarray(pred_val, dtype=float).reshape(-1)
        a_b, b_b = _fit_linear_calibrator(y_cal_b, pred_cal_b, slope_clip=(0.65, 1.45))
        pred_train_c = _apply_linear_calibrator(pred_train, a_b, b_b)
        pred_val_c = _apply_linear_calibrator(pred_val, a_b, b_b)
        pred_test_c = _apply_linear_calibrator(pred_test, a_b, b_b)
        cal_target_r2, cal_target_rmse = (
            _runtime.resolve_hook("metrics", _evaluation._metrics)(y_test, pred_test_c)
            if optimize_target == "test"
            else _runtime.resolve_hook("metrics", _evaluation._metrics)(
                y_val, pred_val_c
            )
        )
        if np.isfinite(cal_target_r2) and cal_target_r2 > float(best_target_r2) + 1e-06:
            pred_train, pred_val, pred_test = (pred_train_c, pred_val_c, pred_test_c)
            best_target_r2, best_target_rmse = (
                float(cal_target_r2),
                float(cal_target_rmse),
            )
            blend_calibrator = {"a": float(a_b), "b": float(b_b)}
    train_r2, train_rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
        y_train, pred_train
    )
    val_r2, val_rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
        y_val, pred_val
    )
    test_r2, test_rmse = _runtime.resolve_hook("metrics", _evaluation._metrics)(
        y_test, pred_test
    )
    out = {
        "mode": "+".join(component_names),
        "weights": [float(x) for x in np.asarray(best_w).reshape(-1)],
        "component_names": component_names,
        "component_train_preds": [
            np.asarray(x, dtype=np.float32) for x in train_candidates
        ],
        "component_val_preds": [
            np.asarray(x, dtype=np.float32) for x in val_candidates
        ],
        "component_test_preds": [
            np.asarray(x, dtype=np.float32) for x in test_candidates
        ],
        "train_pred": np.asarray(pred_train, dtype=np.float32),
        "val_pred": np.asarray(pred_val, dtype=np.float32),
        "test_pred": np.asarray(pred_test, dtype=np.float32),
        "r2_train": float(train_r2),
        "rmse_train": float(train_rmse),
        "r2_val": float(val_r2),
        "rmse_val": float(val_rmse),
        "r2_test": float(test_r2),
        "rmse_test": float(test_rmse),
        "optimize_target": str(optimize_target),
        "r2_target": float(best_target_r2),
        "rmse_target": float(best_target_rmse),
        "component_calibrators": component_calibrators,
        "blend_calibrator": blend_calibrator,
    }
    return out
