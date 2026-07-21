from __future__ import annotations

import sys

from pyspark import cloudpickle

cloudpickle.register_pickle_by_value(sys.modules[__name__])

import math
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Iterator, Sequence

import numpy as np
from pyspark import RDD

MIN_YEAR = 1922.0
MAX_YEAR = 2011.0
YEAR_SPAN = MAX_YEAR - MIN_YEAR
THRESHOLD_COUNT = int(YEAR_SPAN)
DECADE_COUNT = 10
DECADE_CENTERS = np.asarray(
    [1925.0, 1935.0, 1945.0, 1955.0, 1965.0, 1975.0, 1985.0, 1995.0, 2005.0, 2010.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class LossConfig:
    ordinal: float = 0.35
    moe: float = 0.45
    direct: float = 0.05
    decade: float = 0.12
    consistency: float = 0.03
    huber_delta: float = 3.0
    expert_span: float = 8.0
    blend_ordinal: float = 0.20
    blend_moe: float = 0.80
    blend_direct: float = 0.0

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("loss configuration contains non-finite values")
        if min(self.ordinal, self.moe, self.direct, self.decade, self.consistency) < 0.0:
            raise ValueError("loss weights cannot be negative")
        if self.huber_delta <= 0.0 or self.expert_span <= 0.0:
            raise ValueError("Huber delta and expert span must be positive")
        blend_sum = self.blend_ordinal + self.blend_moe + self.blend_direct
        if abs(blend_sum - 1.0) > 1.0e-9:
            raise ValueError("inference blend weights must sum to one")


@dataclass(frozen=True)
class ParameterLayout:
    dimension: int

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("feature dimension must be positive")

    def slices(self) -> dict[str, slice]:
        d = self.dimension
        cursor = 0
        result: dict[str, slice] = {}
        for name, size in (
            ("ordinal_w", d),
            ("ordinal_b", 1),
            ("thresholds", THRESHOLD_COUNT),
            ("gate_w", DECADE_COUNT * d),
            ("gate_b", DECADE_COUNT),
            ("expert_w", DECADE_COUNT * d),
            ("expert_b", DECADE_COUNT),
            ("direct_w", d),
            ("direct_b", 1),
        ):
            result[name] = slice(cursor, cursor + size)
            cursor += size
        return result

    @property
    def size(self) -> int:
        return self.slices()["direct_b"].stop

    def weight_mask(self) -> np.ndarray:
        mask = np.zeros(self.size, dtype=bool)
        slices = self.slices()
        for name in ("ordinal_w", "gate_w", "expert_w", "direct_w"):
            mask[slices[name]] = True
        return mask


def decade_index(years: np.ndarray) -> np.ndarray:
    return np.clip(((years.astype(np.int64) - 1920) // 10), 0, DECADE_COUNT - 1)


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(values / (1.0 - values))


def initialize_parameters(
    layout: ParameterLayout,
    year_counts: Sequence[int],
    seed: int,
) -> np.ndarray:
    if len(year_counts) != THRESHOLD_COUNT + 1:
        raise ValueError("year histogram has the wrong dimension")
    rng = np.random.default_rng(seed)
    parameters = np.zeros(layout.size, dtype=np.float64)
    slices = layout.slices()
    total = max(1, int(sum(year_counts)))
    survival = np.asarray(
        [sum(year_counts[index + 1 :]) / total for index in range(THRESHOLD_COUNT)],
        dtype=np.float64,
    )
    parameters[slices["thresholds"]] = -_logit(survival)
    scale = 0.01 / math.sqrt(layout.dimension)
    for name in ("ordinal_w", "gate_w", "expert_w", "direct_w"):
        parameters[slices[name]] = rng.normal(0.0, scale, slices[name].stop - slices[name].start)
    return parameters


def unpack(parameters: np.ndarray, layout: ParameterLayout) -> dict[str, np.ndarray | float]:
    if parameters.shape != (layout.size,):
        raise ValueError("parameter vector has the wrong shape")
    d = layout.dimension
    slices = layout.slices()
    return {
        "ordinal_w": parameters[slices["ordinal_w"]],
        "ordinal_b": float(parameters[slices["ordinal_b"]][0]),
        "thresholds": parameters[slices["thresholds"]],
        "gate_w": parameters[slices["gate_w"]].reshape(DECADE_COUNT, d),
        "gate_b": parameters[slices["gate_b"]],
        "expert_w": parameters[slices["expert_w"]].reshape(DECADE_COUNT, d),
        "expert_b": parameters[slices["expert_b"]],
        "direct_w": parameters[slices["direct_w"]],
        "direct_b": float(parameters[slices["direct_b"]][0]),
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -35.0, 0.0))
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def huber(error: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(error)
    quadratic = absolute <= delta
    loss = np.where(quadratic, 0.5 * error * error, delta * (absolute - 0.5 * delta))
    gradient = np.where(quadratic, error, delta * np.sign(error))
    return loss, gradient


def forward(
    features: np.ndarray,
    parameters: np.ndarray,
    layout: ParameterLayout,
    config: LossConfig,
) -> dict[str, np.ndarray]:
    values = unpack(parameters, layout)
    ordinal_score = features @ values["ordinal_w"] + values["ordinal_b"]
    ordinal_probability = sigmoid(
        ordinal_score[:, None] - values["thresholds"][None, :]
    )
    ordinal_year = MIN_YEAR + np.sum(ordinal_probability, axis=1)
    gate_logits = features @ values["gate_w"].T + values["gate_b"][None, :]
    gate_probability = softmax(gate_logits)
    expert_raw = features @ values["expert_w"].T + values["expert_b"][None, :]
    expert_tanh = np.tanh(expert_raw)
    expert_years = DECADE_CENTERS[None, :] + config.expert_span * expert_tanh
    moe_year = np.sum(gate_probability * expert_years, axis=1)
    direct_raw = features @ values["direct_w"] + values["direct_b"]
    direct_tanh = np.tanh(direct_raw)
    direct_year = (MIN_YEAR + MAX_YEAR) / 2.0 + (YEAR_SPAN / 2.0) * direct_tanh
    blend_year = (
        config.blend_ordinal * ordinal_year
        + config.blend_moe * moe_year
        + config.blend_direct * direct_year
    )
    return {
        "ordinal_probability": ordinal_probability,
        "ordinal_year": ordinal_year,
        "gate_probability": gate_probability,
        "expert_tanh": expert_tanh,
        "expert_years": expert_years,
        "moe_year": moe_year,
        "direct_tanh": direct_tanh,
        "direct_year": direct_year,
        "blend_year": blend_year,
    }


def batch_gradient(
    features: np.ndarray,
    years: np.ndarray,
    parameters: np.ndarray,
    layout: ParameterLayout,
    config: LossConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    output = forward(features, parameters, layout, config)
    count = years.size
    thresholds = MIN_YEAR + np.arange(THRESHOLD_COUNT, dtype=np.float64)
    ordinal_target = (years[:, None] > thresholds[None, :]).astype(np.float64)
    probability = np.clip(output["ordinal_probability"], 1.0e-8, 1.0 - 1.0e-8)
    ordinal_loss = -np.sum(
        ordinal_target * np.log(probability)
        + (1.0 - ordinal_target) * np.log(1.0 - probability)
    ) / THRESHOLD_COUNT
    moe_loss, moe_gradient = huber(output["moe_year"] - years, config.huber_delta)
    direct_loss, direct_gradient = huber(
        output["direct_year"] - years, config.huber_delta
    )
    true_decade = decade_index(years)
    gate_probability = np.clip(output["gate_probability"], 1.0e-8, 1.0)
    decade_loss = -np.sum(np.log(gate_probability[np.arange(count), true_decade]))
    scale = YEAR_SPAN * YEAR_SPAN
    ordinal_difference = output["ordinal_year"] - output["moe_year"]
    direct_difference = output["ordinal_year"] - output["direct_year"]
    consistency_loss = 0.5 * np.sum(
        ordinal_difference * ordinal_difference + direct_difference * direct_difference
    ) / scale
    loss_parts = np.asarray(
        [
            ordinal_loss,
            float(np.sum(moe_loss)),
            float(np.sum(direct_loss)),
            decade_loss,
            consistency_loss,
        ],
        dtype=np.float64,
    )
    d_ordinal_year = config.consistency * (
        ordinal_difference + direct_difference
    ) / scale
    d_moe_year = config.moe * moe_gradient - config.consistency * ordinal_difference / scale
    d_direct_year = (
        config.direct * direct_gradient - config.consistency * direct_difference / scale
    )
    d_ordinal_logits = (
        config.ordinal * (probability - ordinal_target) / THRESHOLD_COUNT
        + d_ordinal_year[:, None] * probability * (1.0 - probability)
    )
    d_ordinal_score = np.sum(d_ordinal_logits, axis=1)
    d_gate = d_moe_year[:, None] * gate_probability * (
        output["expert_years"] - output["moe_year"][:, None]
    )
    decade_target = np.zeros_like(gate_probability)
    decade_target[np.arange(count), true_decade] = 1.0
    d_gate += config.decade * (gate_probability - decade_target)
    d_expert_raw = (
        d_moe_year[:, None]
        * gate_probability
        * config.expert_span
        * (1.0 - output["expert_tanh"] ** 2)
    )
    d_direct_raw = (
        d_direct_year * (YEAR_SPAN / 2.0) * (1.0 - output["direct_tanh"] ** 2)
    )
    gradient = np.zeros(layout.size, dtype=np.float64)
    slices = layout.slices()
    gradient[slices["ordinal_w"]] = features.T @ d_ordinal_score
    gradient[slices["ordinal_b"]] = np.sum(d_ordinal_score)
    gradient[slices["thresholds"]] = -np.sum(d_ordinal_logits, axis=0)
    gradient[slices["gate_w"]] = (d_gate.T @ features).ravel()
    gradient[slices["gate_b"]] = np.sum(d_gate, axis=0)
    gradient[slices["expert_w"]] = (d_expert_raw.T @ features).ravel()
    gradient[slices["expert_b"]] = np.sum(d_expert_raw, axis=0)
    gradient[slices["direct_w"]] = features.T @ d_direct_raw
    gradient[slices["direct_b"]] = np.sum(d_direct_raw)
    return gradient, loss_parts, count


GradientPartial = tuple[np.ndarray, np.ndarray, int]


def _partition_gradient(
    rows: Iterator[tuple[str, str, int, np.ndarray]],
    parameters: np.ndarray,
    layout: ParameterLayout,
    config: LossConfig,
    batch_size: int,
) -> Iterator[GradientPartial]:
    gradient = np.zeros(layout.size, dtype=np.float64)
    losses = np.zeros(5, dtype=np.float64)
    count = 0
    feature_batch: list[np.ndarray] = []
    year_batch: list[float] = []

    def consume() -> None:
        nonlocal gradient, losses, count
        if not feature_batch:
            return
        partial = batch_gradient(
            np.stack(feature_batch),
            np.asarray(year_batch, dtype=np.float64),
            parameters,
            layout,
            config,
        )
        gradient += partial[0]
        losses += partial[1]
        count += partial[2]
        feature_batch.clear()
        year_batch.clear()

    for _, _, year, features in rows:
        feature_batch.append(features)
        year_batch.append(float(year))
        if len(feature_batch) >= batch_size:
            consume()
    consume()
    if count:
        yield gradient, losses, count


def _merge_gradient(left: GradientPartial, right: GradientPartial) -> GradientPartial:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def distributed_gradient(
    points: RDD[tuple[str, str, int, np.ndarray]],
    parameters: np.ndarray,
    layout: ParameterLayout,
    config: LossConfig,
    l2: float,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float], int]:
    broadcast = points.context.broadcast(parameters)
    try:
        gradient, losses, count = points.mapPartitions(
            lambda rows: _partition_gradient(
                rows, broadcast.value, layout, config, batch_size
            )
        ).treeReduce(_merge_gradient)
    finally:
        broadcast.destroy()
    if count <= 0:
        raise ValueError("gradient batch is empty")
    gradient /= count
    losses /= count
    mask = layout.weight_mask()
    gradient[mask] += l2 * parameters[mask]
    total = (
        config.ordinal * losses[0]
        + config.moe * losses[1]
        + config.direct * losses[2]
        + config.decade * losses[3]
        + config.consistency * losses[4]
        + 0.5 * l2 * float(parameters[mask] @ parameters[mask])
    )
    names = ("ordinal", "moe", "direct", "decade", "consistency")
    metrics = {name: float(value) for name, value in zip(names, losses)}
    metrics["total"] = float(total)
    return gradient, metrics, count


def adam_step(
    parameters: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
    learning_rate: float,
    gradient_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    norm = float(np.linalg.norm(gradient))
    if gradient_clip > 0.0 and norm > gradient_clip:
        gradient = gradient * (gradient_clip / norm)
    beta1, beta2 = 0.9, 0.999
    first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
    second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
    corrected_first = first_moment / (1.0 - beta1**step)
    corrected_second = second_moment / (1.0 - beta2**step)
    parameters = parameters - learning_rate * corrected_first / (
        np.sqrt(corrected_second) + 1.0e-8
    )
    slices = ParameterLayout(
        (parameters.size - THRESHOLD_COUNT - 2 * DECADE_COUNT - 2)
        // (2 * DECADE_COUNT + 2)
    ).slices()
    thresholds = parameters[slices["thresholds"]]
    thresholds = np.maximum.accumulate(np.clip(thresholds, -12.0, 12.0))
    thresholds += np.arange(THRESHOLD_COUNT) * 1.0e-7
    parameters[slices["thresholds"]] = thresholds
    if not np.all(np.isfinite(parameters)):
        raise FloatingPointError("optimizer produced non-finite parameters")
    return parameters, first_moment, second_moment, norm


def prediction_partition(
    rows: Iterator[tuple[str, str, int, np.ndarray]],
    parameters: np.ndarray,
    layout: ParameterLayout,
    config: LossConfig,
    batch_size: int,
) -> Iterator[tuple[str, str, int, float, float, float, float]]:
    buffer: list[tuple[str, str, int, np.ndarray]] = []

    def predict_buffer():
        if not buffer:
            return []
        features = np.stack([row[3] for row in buffer])
        output = forward(features, parameters, layout, config)
        result = [
            (
                row[0],
                row[1],
                row[2],
                float(output["ordinal_year"][index]),
                float(output["moe_year"][index]),
                float(output["direct_year"][index]),
                float(output["blend_year"][index]),
            )
            for index, row in enumerate(buffer)
        ]
        buffer.clear()
        return result

    for row in rows:
        buffer.append(row)
        if len(buffer) >= batch_size:
            yield from predict_buffer()
    yield from predict_buffer()
