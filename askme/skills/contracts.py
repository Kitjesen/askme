"""Code-defined skill contracts and OpenAPI generation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class SkillParameter:
    """A single structured parameter exposed by a skill contract."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: tuple[str, ...] = ()

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.default not in (None, ""):
            schema["default"] = self.default
        return schema


@dataclass(frozen=True)
class SkillContract:
    """Code-level contract from which docs, MCP views, and OpenAPI are generated."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    safety_level: str = "normal"
    execution: str = "skill_executor"
    tags: tuple[str, ...] = ()
    parameters: tuple[SkillParameter, ...] = ()
    confirm_before_execute: bool = False
    source: str = "code"

    def with_fallbacks(
        self,
        *,
        description: str = "",
        version: str = "",
        safety_level: str = "",
        execution: str = "",
        tags: Iterable[str] = (),
        confirm_before_execute: bool = False,
    ) -> "SkillContract":
        """Fill unset contract metadata from legacy/loaded skill metadata."""
        fallback_tags = tuple(tag for tag in tags if tag)
        return replace(
            self,
            description=self.description or description,
            version=self.version or version or "1.0.0",
            safety_level=self.safety_level or safety_level or "normal",
            execution=self.execution or execution or "skill_executor",
            tags=self.tags or fallback_tags,
            confirm_before_execute=self.confirm_before_execute or confirm_before_execute,
        )

    def request_schema(self) -> dict[str, Any]:
        """JSON Schema for the execute request body."""
        properties: dict[str, Any] = {
            "user_input": {
                "type": "string",
                "description": "Natural-language instruction or task context.",
            }
        }
        required = ["user_input"]
        for parameter in self.parameters:
            properties[parameter.name] = parameter.json_schema()
            if parameter.required:
                required.append(parameter.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def openapi_path_item(self) -> dict[str, Any]:
        """Return the OpenAPI path item for executing this skill."""
        return {
            "post": {
                "summary": f"Execute {self.name}",
                "description": self.description or f"Execute the {self.name} skill.",
                "tags": ["Skills"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": self.request_schema(),
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Skill execution result.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "skill_name": {"type": "string"},
                                        "reply": {"type": "string"},
                                        "status": {"type": "string"},
                                    },
                                    "required": ["skill_name", "reply", "status"],
                                }
                            }
                        },
                    }
                },
                "x-askme-skill": {
                    "name": self.name,
                    "version": self.version,
                    "safety_level": self.safety_level,
                    "execution": self.execution,
                    "confirm_before_execute": self.confirm_before_execute,
                    "contract_source": self.source,
                },
            }
        }

    def summary(self) -> dict[str, Any]:
        """Compact catalog entry."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "safety_level": self.safety_level,
            "execution": self.execution,
            "tags": list(self.tags),
            "parameter_count": len(self.parameters),
            "contract_source": self.source,
        }


class SkillContractRegistry:
    """In-memory registry for code-defined skill contracts."""

    def __init__(self) -> None:
        self._contracts: dict[str, SkillContract] = {}

    def register(self, contract: SkillContract) -> SkillContract:
        self._contracts[contract.name] = contract
        return contract

    def get(self, name: str) -> SkillContract | None:
        return self._contracts.get(name)

    def all(self) -> list[SkillContract]:
        return [self._contracts[name] for name in sorted(self._contracts)]


_REGISTRY = SkillContractRegistry()


def register_skill_contract(contract: SkillContract) -> SkillContract:
    """Register a concrete contract instance."""
    return _REGISTRY.register(contract)


def registered_skill_contracts() -> dict[str, SkillContract]:
    """Return a copy of the current contract map."""
    return {contract.name: contract for contract in _REGISTRY.all()}


def skill_contract(
    *,
    name: str,
    description: str = "",
    version: str = "1.0.0",
    safety_level: str = "normal",
    execution: str = "skill_executor",
    tags: Iterable[str] = (),
    parameters: Iterable[SkillParameter] = (),
    confirm_before_execute: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator used to declare a code-defined skill contract."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        contract = SkillContract(
            name=name,
            description=description,
            version=version,
            safety_level=safety_level,
            execution=execution,
            tags=tuple(tag for tag in tags if tag),
            parameters=tuple(parameters),
            confirm_before_execute=confirm_before_execute,
            source="code",
        )
        register_skill_contract(contract)
        setattr(func, "__askme_skill_contract__", contract)
        return func

    return decorator


def build_skills_openapi(
    contracts: Iterable[SkillContract],
    *,
    title: str = "Askme Skill Runtime API",
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Generate a minimal OpenAPI document from skill contracts."""
    ordered_contracts = sorted(contracts, key=lambda contract: contract.name)
    paths: dict[str, Any] = {
        "/api/v1/skills": {
            "get": {
                "summary": "List available skills",
                "tags": ["Skills"],
                "responses": {
                    "200": {
                        "description": "Available skills.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "skills": {
                                            "type": "array",
                                            "items": {"type": "object"},
                                        },
                                        "count": {"type": "integer"},
                                    },
                                    "required": ["skills", "count"],
                                }
                            }
                        },
                    }
                },
            }
        }
    }
    for contract in ordered_contracts:
        paths[f"/api/v1/skills/{contract.name}/execute"] = contract.openapi_path_item()
    return {
        "openapi": "3.1.0",
        "info": {
            "title": title,
            "version": version,
        },
        "tags": [
            {
                "name": "Skills",
                "description": "Generated from askme skill contracts.",
            }
        ],
        "paths": paths,
    }
