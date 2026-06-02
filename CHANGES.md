# Changelog

## 0.0.12

### Fixed
- pandas 3.0 compatibility in `parse_string_data_column`: replace the deprecated
  `Series.replace(to_replace=-1, method='ffill', inplace=True)` (which also relied
  on chained-assignment that silently no-ops under Copy-on-Write) with
  `df_data[flag] = df_data[flag].replace(-1, np.nan).ffill()` followed by filling
  the remaining leading sentinels with 0. Removes two `FutureWarning`s and preserves
  the original forward-fill-over-sentinel behaviour.

## 0.0.11

### Changed
- Migrate off `pkg_resources` (removed in setuptools 81) to `importlib.resources`
  for loading the bundled CSV metadata (`methods.csv`, `stop_reasons.csv`, `flags.csv`).

No other functional changes since 0.0.10.
