# Text Encoding Hygiene

This repository stores source and docs as UTF-8. On Windows, mojibake in the
terminal often means the console is decoding UTF-8 output through an older code
page such as CP936. It does not always mean the file is corrupted.

## Quick Check

Run the repository scanner:

```powershell
python scripts/check_text_encoding.py
```

The scanner fails on:

- UTF-8 decode errors.
- The Unicode replacement character `U+FFFD`.
- Common mojibake markers such as `\u9239\u20ac`, `\u9225`, `\u922b`,
  `\u00e2\u20ac`, and `\u00ef\u00bf\u00bd`.

It skips generated and local state directories such as `.git`, `.omx`,
`.omc`, `.claude`, `.venv-scientist`, `.tmp`, `tests/tmp`, and
`data/pytest-tmp`.

## Windows Console

For a single PowerShell session:

```powershell
.\scripts\enable_utf8_console.ps1
```

Equivalent commands:

```powershell
chcp 65001
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

This changes only the current shell. Avoid committing text that was copied from
a mojibake terminal transcript; verify with the scanner first.
