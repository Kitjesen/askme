# Askme Docker Surface

`docker/` contains container build and Compose assets for Askme + ZeroClaw.
Product code remains under `askme/`; this directory only packages and wires the
runtime for local or production container deployment.

## Files

| File | Purpose |
| --- | --- |
| `.env.example` | Environment variable template. Copy to `docker/.env` for local Compose runs. |
| `docker-compose.yml` | Base Askme + ZeroClaw Compose stack. |
| `docker-compose.prod.yml` | Production override for resource limits, logging, and stricter runtime flags. |
| `Dockerfile.askme` | Askme service image. |
| `Dockerfile.zeroclaw` | ZeroClaw gateway image. |
| `docker-entrypoint.sh` | Askme container entrypoint. |

## Commands

From the repository root:

```powershell
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
```

Production overlay:

```powershell
docker compose --env-file docker/.env -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

Use `deploy/quickstart.sh` or `deploy/quickstart.bat` when you want the helper
to create `docker/.env` from local config first.

## Verification

For docs and static checks:

```powershell
python -m pytest tests/test_deploy_paths.py tests/test_scripts_structure.py tests/test_scripts_static.py -q
```
