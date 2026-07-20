# Feature Tests

- `test_contract.py`: checks dimensions, group disjointness, T90 order, and deterministic hashes.
- `test_t90.py`: checks that T90 projection preserves audit fields, values, and exact order.
- `test_full_tabular.py`: checks global category cleaning and valid, tolerated, and invalid fade ratios.

The full-data validator performs source-to-output value and coverage checks separately from these unit tests.
