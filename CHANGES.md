# Changelog

## 0.0.11

### Changed
- Migrate off `pkg_resources` (removed in setuptools 81) to `importlib.resources`
  for loading the bundled CSV metadata (`methods.csv`, `stop_reasons.csv`, `flags.csv`).

No other functional changes since 0.0.10.
