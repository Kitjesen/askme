# askme.interaction

Compatibility facade for legacy imports.

New code should import interaction logic from `askme.robot_interaction`.
Do not add new implementation logic here.

Current role:

- keep `askme.interaction.*` import paths working;
- re-export intent routing, routing policy, observability, and scenario intent
  helpers from `askme.robot_interaction`;
- give old tests and integrations time to migrate.

