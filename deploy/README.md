# Askme Deploy Surface

`deploy/` contains install-time and customer handoff assets. It is not a
product-code package and should not grow runtime business logic.

## Directory Map

| Path | Owns | Notes |
| --- | --- | --- |
| `askme.service` | Production systemd unit for the edge runtime. | Defaults to `/home/sunrise/data/inovxio/askme`; override through `/etc/default/askme`. |
| `install.sh` | S100P/Sunrise systemd install helper. | Requires operator review before running on a device. |
| `quickstart.sh` | Local Linux/macOS Askme + ZeroClaw start helper. | Uses `docker/docker-compose.yml` explicitly. |
| `quickstart.bat` | Local Windows Askme + ZeroClaw start helper. | Avoids machine-specific absolute paths. |
| `site-profiles/` | Default field site profiles used by product tests and demos. | Keep stable paths; many tests and demos reference these files. |
| `customer-project-templates/` | Customer project templates. | Field customer project routes read from this root by default. |
| `delivery-resources/` | Shared delivery resource registry. | Used by field delivery package generation. |
| `security/` | Deployment hardening notes and operational security guidance. | Documentation only. |

## Placement Rules

1. Service units, installer scripts, and customer deployment assets belong here.
2. Runtime implementation remains in `askme/`.
3. Long-running bridge/service launchers that are not install assets belong
   under `scripts/runtime/`.
4. Local developer sync helpers belong under `scripts/dev/`.
5. Generated reports, customer dossiers, logs, and screenshots belong under
   `artifacts/` or customer-specific deploy artifact roots, not beside scripts.

## Verification

For deploy path changes:

```powershell
python -m pytest tests/test_deploy_paths.py -q
```

For field customer profile/template changes:

```powershell
python -m pytest tests/test_field_site_profile.py tests/test_field_http.py -q --tb=short
```
