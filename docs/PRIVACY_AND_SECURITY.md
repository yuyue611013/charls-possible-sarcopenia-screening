# Privacy and Security

This release was curated to exclude:

- Raw CHARLS microdata.
- Processed row-level participant datasets.
- Model-input tables containing individual records.
- Row-level prediction files.
- Trained model objects.
- Participant identifier lists.
- Manuscript drafts.
- Local absolute paths and private machine-specific paths.
- Secrets, API keys, tokens, passwords, and private keys.

The public-release checker in `tools/check_public_release_v1.py` inspects only the release folder. It fails on common raw-data/model-object extensions, personal absolute paths, common credential patterns, and unexpected CSV files outside the aggregate-output allowlist.

## Metadata Synchronisation Note

The v1.0.0 metadata synchronisation changed public-facing title and framing only. It did not add raw data, participant-level files, prediction files, model objects, or manuscript drafts.
