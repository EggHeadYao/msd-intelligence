from __future__ import annotations

from typing import Any


K0 = "k0"
K1 = "k1"
K2 = "k2"
K3 = "k3"
KEY_CONTRACTS = (K0, K1, K2, K3)

KEY_SIN_COLUMN = "key_sin"
KEY_COS_COLUMN = "key_cos"
KEY_UNKNOWN_COLUMN = "key_unknown"

ENCODING_NAMES = {
    K0: "no_key",
    K1: "one_hot",
    K2: "chromatic_circle",
    K3: "circle_of_fifths",
}


def require_key_contract(contract: str) -> str:
    if contract not in KEY_CONTRACTS:
        raise ValueError(f"Unsupported key contract: {contract}")
    return contract


def key_one_hot_column(value: int) -> str:
    if value not in range(12):
        raise ValueError(f"Key value must be in [0, 11]: {value}")
    return f"key_{value}"


def key_feature_columns(contract: str) -> tuple[str, ...]:
    contract = require_key_contract(contract)
    if contract == K0:
        return ()
    if contract == K1:
        return (*(key_one_hot_column(value) for value in range(12)), KEY_UNKNOWN_COLUMN)
    return KEY_SIN_COLUMN, KEY_COS_COLUMN, KEY_UNKNOWN_COLUMN


def key_contract_metadata(contract: str) -> dict[str, Any]:
    contract = require_key_contract(contract)
    return {
        "id": contract,
        "encoding": ENCODING_NAMES[contract],
        "columns": list(key_feature_columns(contract)),
    }
