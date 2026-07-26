            "training_universe": "set_a" if stage == "tuning" else "a_b_remaining",
            "fill_values": {name: float(value) for name, value in fill_values.items()},
            "means": [float(value) for value in means],
            "stds": [float(value) for value in stds],
            "constant_features": list(constant),
            "constant_feature_scale": "effective_std_1_with_zero_model_weight",
        },
        scaler_path,
    )
    write_json_atomic(
        {
            **common,
            "artifact_type": "ranker_coefficients",
            "model_type": "logistic_regression",
            "model_version": "full-merlin-lr-v1",
            "elastic_net_param": 0.0,
            "reg_param": float(reg_param),
            "max_iter": 100,
            "tol": 1e-6,
            "fit_intercept": True,
            "standardization": True,
            "coefficients": [float(value) for value in coefficients],
            "intercept": float(intercept),
        },
        coefficients_path,
    )
    manifest = {
        **common,
        "artifact_type": "ranker_training",
        "artifact_version": RANKER_TRAINING_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "stage": stage,
        "selected_reg_param": float(reg_param),
        "converged": converged,
        "iterations": int(iterations),
        "selection": dict(selection),
        "constant_features": list(constant),
        "artifact_hashes": {
            path.name: sha256_path(path)
            for path in (schema_path, scaler_path, coefficients_path)
        },
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest
