from __future__ import annotations

from html.parser import HTMLParser

from askme.pipeline.field.customer_project_package_html import (
    _render_customer_project_acceptance_dossier_html,
    _render_customer_project_proposal_bundle_html,
)

MOJIBAKE_FRAGMENTS = (
    "浜や",
    "涓婄",
    "椤圭",
    "楠屾",
    "璇佹",
    "闃诲",
    "缂哄",
    "娴溿",
    "妤犲",
    "鐠囦",
)


class _TagCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.starts: dict[str, int] = {}
        self.ends: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        self.starts[tag] = self.starts.get(tag, 0) + 1

    def handle_endtag(self, tag: str) -> None:
        self.ends[tag] = self.ends.get(tag, 0) + 1


def _assert_customer_html_is_structured(html: str) -> None:
    parser = _TagCounter()
    parser.feed(html)

    assert parser.starts.get("h2", 0) == parser.ends.get("h2", 0)
    assert "\ufffd" not in html
    for fragment in MOJIBAKE_FRAGMENTS:
        assert fragment not in html


def test_acceptance_dossier_html_is_customer_readable_and_escaped() -> None:
    html = _render_customer_project_acceptance_dossier_html(
        {
            "customer": {
                "customer_name": "<b>梵木客户</b>",
                "project_id": "fanmu-demo",
            },
            "site": {"site_id": "fanmu"},
            "manifest": {"payload_sha256": "a" * 64},
            "overall_status": "ready",
            "customer_status": "可进入现场试点验收",
            "release_claim": "仅承诺试点能力，不承诺无人值守生产上线",
            "field_readiness": {
                "status": "ready",
                "delivery_brief": {"customer_status": "可进入现场试点验收"},
            },
            "launch_readiness": {
                "launch_stage": "onsite_trial",
                "overall_status": "ready",
                "production_ready": False,
                "next_step": "安排现场验收",
                "gates": [{"label": "<上线准入>", "status": "ready"}],
            },
            "delivery_workflow": {
                "steps": [{"label": "上传知识", "status": "ready"}],
            },
            "delivery_chain": {
                "steps": [{"label": "客户交付链路", "status": "ready"}],
            },
            "gates": [{"label": "证据齐全", "status": "ready"}],
            "evidence_inventory": [
                {"path": "artifacts/<evidence>.json", "exists": True, "sha256": "b" * 64}
            ],
        }
    )

    assert "AskMe 客户验收资料包" in html
    assert "交付结论" in html
    assert "上线准入" in html
    assert "客户交付链路" in html
    assert "验收门禁" in html
    assert "证据文件清单" in html
    assert "下一步" in html
    assert "&lt;b&gt;梵木客户&lt;/b&gt;" in html
    assert "<b>梵木客户</b>" not in html
    assert "&lt;上线准入&gt;" in html
    _assert_customer_html_is_structured(html)


def test_proposal_bundle_html_keeps_launch_gate_copy_readable_and_escaped() -> None:
    html = _render_customer_project_proposal_bundle_html(
        {
            "customer": {"customer_name": "梵木创艺园", "project_name": "<首园区试点>"},
            "site": {"name": "成都梵木"},
            "customer_project_package": {
                "manifest": {
                    "managed_object_count": 6,
                    "acceptance_overall_status": "ready",
                    "payload_sha256": "c" * 64,
                }
            },
            "acceptance_dossier": {
                "overall_status": "ready",
                "manifest": {"payload_sha256": "d" * 64},
                "gates": [{"label": "客户可读验收项", "status": "ready"}],
            },
            "launch_readiness": {
                "launch_stage": "onsite_trial",
                "customer_status": "可进入现场试点验收",
                "release_claim": "仅承诺试点能力",
                "gates": [{"label": "<上线准入>", "status": "ready"}],
            },
            "proposal_insert": {
                "section_title": "可复用能力包",
                "customer_message": "面向园区巡检交付",
                "safe_claims": ["支持指路和异常上报"],
                "delivery_boundaries": ["不承诺开放域闲聊"],
            },
            "customer_readable_delivery": {
                "delivery_chain": {
                    "steps": [{"label": "客户交付链路", "status": "ready"}],
                }
            },
            "approved_template_release_bundle": {},
            "manifest": {"payload_sha256": "e" * 64},
        }
    )

    assert "AskMe 客户项目提案包" in html
    assert "上线准入" in html
    assert "客户交付范围" in html
    assert "客户交付链路" in html
    assert "已发布模板说明" in html
    assert "验收门禁" in html
    assert "交付边界" in html
    assert "&lt;首园区试点&gt;" in html
    assert "&lt;上线准入&gt;" in html
    assert "<首园区试点>" not in html
    assert "<strong>上线准入</strong>" in html
    _assert_customer_html_is_structured(html)
