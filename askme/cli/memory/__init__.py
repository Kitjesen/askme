"""Memory CLI commands — import and query robot long-term memory/RAG knowledge."""

from __future__ import annotations

import asyncio
from typing import Any

from askme.cli.utils import _emit_payload


def _handle_memory_command(args: Any) -> None:
    """Handle the 'memory' command group: import, search."""
    if args.memory_command is None:
        raise SystemExit("Missing memory command. Use: askme memory import|search")

    if args.memory_command == "import":
        from askme.memory.retrieval.importer import import_knowledge_file

        payload = asyncio.run(
            import_knowledge_file(
                args.path,
                source=args.source or None,
                category=args.category or None,
                dry_run=bool(args.dry_run),
            )
        ).to_dict()
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print(  # noqa: T201
                "imported={imported} parsed={parsed} skipped={skipped} errors={errors} source={source}".format(
                    **payload
                )
            )
        if payload.get("errors"):
            raise SystemExit(1)
        return

    if args.memory_command == "search":
        from askme.memory.retrieval.bridge import MemoryBridge

        query = " ".join(args.query)
        bridge = MemoryBridge()
        if args.timeout and args.timeout > 0:
            bridge._retrieve_timeout = float(args.timeout)
        text = asyncio.run(bridge.retrieve(query))
        payload = {
            "query": query,
            "results": [
                line.strip().lstrip("- ").strip()
                for line in text.splitlines()
                if line.strip()
            ],
            "rag": bridge.health(),
        }
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            if not payload["results"]:
                print("No matching memories found")  # noqa: T201
            for item in payload["results"]:
                print(f"- {item}")  # noqa: T201
        return

    raise SystemExit(f"Unknown memory command: {args.memory_command}")
