"""Blueprint API response contracts for delivery handoff surfaces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BlueprintDeliveryPackage(BaseModel):
    """Customer-facing runtime blueprint handoff package."""

    model_config = ConfigDict(extra="allow")

    package_id: str = Field(min_length=1)
    blueprint: str = Field(min_length=1)
    status: str = Field(min_length=1)
    customer_status: str = Field(min_length=1)
    customer_next_step: str = Field(min_length=1)
    acceptance_boundary: str = Field(min_length=1)
    delivery_actions: list[str] = Field(min_length=1)


class BlueprintCatalogItem(BaseModel):
    """One runtime blueprint item in the product catalog."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    title: str | None = None
    product_stage: str | None = None
    customer_visible: bool | None = None
    readiness: dict[str, Any] = Field(default_factory=dict)
    delivery_package: BlueprintDeliveryPackage | None = None


class BlueprintCatalogResponse(BaseModel):
    """Catalog of runtime blueprints used by Dashboard and delivery tools."""

    model_config = ConfigDict(extra="allow")

    summary: dict[str, Any] = Field(default_factory=dict)
    items: list[BlueprintCatalogItem] = Field(default_factory=list)


class BlueprintDetailResponse(BaseModel):
    """Single blueprint response resolved by canonical name or public alias."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    blueprint: BlueprintCatalogItem
    policy: dict[str, Any] = Field(default_factory=dict)


class BlueprintDeliveryPackageResponse(BaseModel):
    """Narrow response for exporting one customer handoff package."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    blueprint: str = Field(min_length=1)
    delivery_package: BlueprintDeliveryPackage
    policy: dict[str, Any] = Field(default_factory=dict)
