# askme.data

Package-local data parking area.

Runtime state and customer data belong in the repository-root `data/`
directory. New code should not import from `askme.data`, and this directory
must not contain Python modules.

Existing files here are retained only to avoid deleting local data during the
layout cleanup.

