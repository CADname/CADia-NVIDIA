# Source provenance

This GitHub package keeps the executable CADia source aligned with the provided deployed-server snapshot.

The following runtime/deployment areas are copied from the server snapshot without GitHub-only behavior changes:

- `src/`
- `web/`
- `tools/`
- `scripts/`
- `tests/`
- `deploy/`
- `Dockerfile`
- `docker-compose.yml`
- Python requirement/project/test configuration files
- `.dockerignore` and `.env.example`

Only repository presentation material such as the NVIDIA/Nebius README, judging documentation, license metadata, and saved evidence is packaging material. Secrets, local `.env` files, caches, patch backups, rejected patches, logs, and generated dependency/build directories are intentionally excluded.

`SERVER_RUNTIME_SHA256.txt` records hashes for the mirrored runtime files.
