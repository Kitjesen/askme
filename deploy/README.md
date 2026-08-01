# Askme Deploy Surface

`deploy/` contains install-time and customer handoff assets. It is not a
product-code package and should not grow runtime business logic.

## Directory Map

| Path | Owns | Notes |
| --- | --- | --- |
| `askme.service` | Production systemd unit for the edge runtime. | Defaults to `/home/sunrise/data/inovxio/askme`; override through `/etc/default/askme`. |
| `install.sh` | S100P/Sunrise systemd install helper. | Requires operator review before running on a device. |
| `quickstart.sh` | Linux/macOS helper; Docker is Linux-edge only. | Docker commands validate `/dev/snd` and resolve `ASKME_AUDIO_GID` before starting LiteLLM; macOS may use local commands. |
| `quickstart.bat` | Windows local-development helper. | Docker commands fail before starting services because the edge image requires Linux `/dev/snd`; local commands remain available. |
| `site-profiles/` | Default field site profiles used by product tests and demos. | Keep stable paths; many tests and demos reference these files. |
| `customer-project-templates/` | Customer project templates. | Field customer project routes read from this root by default. |
| `delivery-resources/` | Shared delivery resource registry. | Used by field delivery package generation. |
| `security/` | Deployment hardening notes and operational security guidance. | Documentation only. |

`docker-zeroclaw` and `local-zeroclaw` are explicit
experimental commands. They start a model-routed v0.1.7 gateway but do not
supply the missing ZeroClaw-to-AskMe MCP connector and must not be used as
integration acceptance evidence. The local helper maps the scoped key only to
the gateway child environment; it does not write the compatibility variable to
a file or argv.

## Local Process Ownership

The local quickstarts record processes they create in
`data/runtime/askme-local.pid` and, for the experimental command,
`data/runtime/zeroclaw-local.pid`. Each record includes the PID and process
creation identity to reject PID reuse. Repeated `local` starts reuse a
still-running recorded process instead of launching a duplicate. If AskMe or
ZeroClaw exits outside the helper, the next start or stop treats the record as
a stale PID file and removes it only after checking that the recorded process
is absent or does not match the expected command and creation identity.

`stop` is idempotent when either PID file is absent. It signals only a PID whose
current command still matches the corresponding recorded process, and never
stops a process by image name or command-pattern search. This keeps another
AskMe checkout or ZeroClaw instance on the same host outside the quickstart's
lifecycle.

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
