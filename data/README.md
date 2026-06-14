# Data Directory

CHARLS data are not distributed with this repository.

Users must obtain authorised access from the China Health and Retirement Longitudinal Study (CHARLS) and comply with all CHARLS data-use terms. Participant-level data must not be committed to GitHub.

Suggested local-only layout after authorisation:

- `data/authorised_charls_wave1_input.csv`
- `data/authorised_charls_wave3_input.csv`

These names are examples only. If your authorised local files use different names, update a private copy of `config/path_config.example.json`. The `data/` directory is ignored by Git except for this README.

Expected data families include participant identifier, wave indicator, age, sex, handgrip strength components, five-chair-stand time, and candidate predictor domains documented in `docs/INPUT_REQUIREMENTS.md`.
