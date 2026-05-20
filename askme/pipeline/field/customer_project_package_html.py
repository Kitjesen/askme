"""Customer-project package and dossier HTML renderers."""

from __future__ import annotations

import html
from typing import Any

from askme.pipeline.field.customer_project_template_support import _mapping, _string_list


def _render_customer_project_acceptance_dossier_html(dossier: dict[str, Any]) -> str:
    customer = _mapping(dossier.get("customer"))
    site = _mapping(dossier.get("site"))
    manifest = _mapping(dossier.get("manifest"))
    readiness = _mapping(dossier.get("field_readiness"))
    launch_readiness = _mapping(dossier.get("launch_readiness"))
    delivery_brief = _mapping(readiness.get("delivery_brief"))
    delivery_workflow = _mapping(dossier.get("delivery_workflow"))
    delivery_chain = _mapping(dossier.get("delivery_chain"))
    gates = dossier.get("gates") if isinstance(dossier.get("gates"), list) else []
    launch_gates = (
        launch_readiness.get("gates")
        if isinstance(launch_readiness.get("gates"), list)
        else []
    )
    workflow_steps = (
        delivery_workflow.get("steps")
        if isinstance(delivery_workflow.get("steps"), list)
        else []
    )
    delivery_chain_steps = (
        delivery_chain.get("steps")
        if isinstance(delivery_chain.get("steps"), list)
        else []
    )
    evidence = dossier.get("evidence_inventory") if isinstance(dossier.get("evidence_inventory"), list) else []
    env_missing = dossier.get("env_missing") if isinstance(dossier.get("env_missing"), list) else []
    warnings = dossier.get("warnings") if isinstance(dossier.get("warnings"), list) else []
    errors = dossier.get("errors") if isinstance(dossier.get("errors"), list) else []
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    next_actions = readiness.get("next_actions") if isinstance(readiness.get("next_actions"), list) else []
    workflow_rows = "\n".join(_dossier_workflow_row(_mapping(item)) for item in workflow_steps)
    if not workflow_rows:
        workflow_rows = "<tr><td colspan=\"4\">No delivery workflow was recorded.</td></tr>"
    delivery_chain_rows = "\n".join(_dossier_workflow_row(_mapping(item)) for item in delivery_chain_steps)
    if not delivery_chain_rows:
        delivery_chain_rows = "<tr><td colspan=\"4\">No customer delivery chain was recorded.</td></tr>"
    gate_rows = "\n".join(_dossier_gate_row(_mapping(item)) for item in gates)
    launch_gate_rows = "\n".join(_dossier_gate_row(_mapping(item)) for item in launch_gates)
    if not launch_gate_rows:
        launch_gate_rows = "<tr><td colspan=\"4\">No launch readiness gates recorded.</td></tr>"
    evidence_rows = "\n".join(_dossier_evidence_row(_mapping(item)) for item in evidence)
    issue_rows = "\n".join(
        f"<li>{_h(item)}</li>"
        for item in [*errors, *blockers, *warnings[:10]]
    ) or "<li>No blocking evidence recorded in this dossier.</li>"
    next_action_rows = "\n".join(f"<li>{_h(item)}</li>" for item in next_actions) or "<li>No next action recorded.</li>"
    missing_env_rows = "\n".join(
        f"<li><strong>{_h(_mapping(item).get('env_name'))}</strong> - {_h(_mapping(item).get('purpose'))}</li>"
        for item in env_missing
    ) or "<li>No missing environment variable recorded.</li>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AskMe 客户验收资料包 - {_h(customer.get('project_id') or site.get('site_id'))}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #13211c;
      --muted: #60706b;
      --line: #dfe8e4;
      --soft: #f4f8f6;
      --ok: #0f8a57;
      --warn: #a76606;
      --bad: #b3261e;
      --accent: #0c6b4f;
    }}
    body {{
      margin: 0;
      background: #eef5f2;
      color: var(--ink);
      font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 28px 56px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      padding: 28px;
      background: linear-gradient(135deg, #ffffff, #e7f3ee);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.2; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; }}
    section {{
      margin-top: 18px;
      padding: 22px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .meta, .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .card, .metric {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--soft);
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-weight: 700; word-break: break-word; }}
    .status {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      font-weight: 700;
      border: 1px solid currentColor;
    }}
    .ok {{ color: var(--ok); }}
    .manual_check, .ready_for_lab, .warn {{ color: var(--warn); }}
    .blocked, .err {{ color: var(--bad); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; vertical-align: top; padding: 10px 8px; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-size: 12px; }}
    code {{ word-break: break-all; }}
    .boundary {{
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      background: #eef8f4;
      border-radius: 10px;
    }}
    @media print {{
      body {{ background: #fff; }}
      main {{ padding: 0; }}
      header, section {{ break-inside: avoid; border-radius: 0; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="label">AskMe 客户验收资料包</div>
      <h1>{_h(customer.get('customer_name') or customer.get('customer_id') or 'Customer Project')}</h1>
      <p>{_h(customer.get('project_name') or customer.get('project_id') or site.get('name') or '')}</p>
    </div>
    <div>
      <div class="label">总体状态</div>
      <div class="status {_status_class(dossier.get('overall_status'))}">{_h(dossier.get('overall_status'))}</div>
    </div>
  </header>

  <section>
    <h2>交付结论</h2>
    <div class="boundary">{_h(dossier.get('customer_status') or delivery_brief.get('customer_status') or '')}</div>
    <p>{_h(dossier.get('handoff_boundary') or '')}</p>
    <p>{_h(dossier.get('release_claim') or delivery_brief.get('release_claim') or '')}</p>
  </section>

  <section>
    <h2>上线准入</h2>
    <div class="meta">
      {_metric('上线阶段', launch_readiness.get('launch_stage'))}
      {_metric('准入状态', launch_readiness.get('overall_status'))}
      {_metric('是否可生产上线', launch_readiness.get('production_ready'))}
      {_metric('下一步', launch_readiness.get('next_step'))}
    </div>
    <p>{_h(launch_readiness.get('customer_status'))}</p>
    <p>{_h(launch_readiness.get('release_claim'))}</p>
    <table>
      <thead><tr><th>门禁</th><th>状态</th><th>证据</th><th>下一步</th></tr></thead>
      <tbody>{launch_gate_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>项目摘要</h2>
    <div class="meta">
      {_metric('客户 ID', customer.get('customer_id'))}
      {_metric('项目 ID', customer.get('project_id'))}
      {_metric('现场 ID', site.get('site_id'))}
      {_metric('行业', customer.get('industry'))}
      {_metric('现场就绪', readiness.get('status'))}
      {_metric('清单 SHA-256', str(manifest.get('payload_sha256') or '')[:24])}
    </div>
  </section>

  <section>
    <h2>交付流程</h2>
    <table>
      <thead><tr><th>步骤</th><th>状态</th><th>证据</th><th>下一步</th></tr></thead>
      <tbody>{workflow_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>客户交付链路</h2>
    <table>
      <thead><tr><th>步骤</th><th>状态</th><th>证据</th><th>下一步</th></tr></thead>
      <tbody>{delivery_chain_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>验收门禁</h2>
    <table>
      <thead><tr><th>门禁</th><th>状态</th><th>证据</th><th>下一步</th></tr></thead>
      <tbody>{gate_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>证据文件清单</h2>
    <table>
      <thead><tr><th>路径</th><th>状态</th><th>大小</th><th>SHA-256</th></tr></thead>
      <tbody>{evidence_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>阻塞项和风险</h2>
    <ul>{issue_rows}</ul>
  </section>

  <section>
    <h2>缺失部署配置</h2>
    <ul>{missing_env_rows}</ul>
  </section>

  <section>
    <h2>下一步</h2>
    <ul>{next_action_rows}</ul>
  </section>

  <section>
    <h2>完整性清单</h2>
    <div class="cards">
      {_metric('证据数量', manifest.get('evidence_count'))}
      {_metric('缺失证据', manifest.get('evidence_missing_count'))}
      {_metric('签名算法', manifest.get('signature_alg') or 'unsigned')}
      {_metric('签名密钥', manifest.get('signature_key_id') or '-')}
    </div>
  </section>
</main>
</body>
</html>
"""


def _dossier_gate_row(gate: dict[str, Any]) -> str:
    status = str(gate.get("status") or "unknown")
    return (
        "<tr>"
        f"<td><strong>{_h(gate.get('label') or gate.get('gate_id'))}</strong></td>"
        f"<td><span class=\"status {_status_class(status)}\">{_h(status)}</span></td>"
        f"<td>{_h(gate.get('evidence'))}</td>"
        f"<td>{_h(gate.get('next_step'))}</td>"
        "</tr>"
    )


def _dossier_workflow_row(step: dict[str, Any]) -> str:
    status = str(step.get("status") or "unknown")
    return (
        "<tr>"
        f"<td><strong>{_h(step.get('label') or step.get('step_id'))}</strong></td>"
        f"<td><span class=\"status {_status_class(status)}\">{_h(status)}</span></td>"
        f"<td>{_h(step.get('evidence'))}</td>"
        f"<td>{_h(step.get('next_step'))}</td>"
        "</tr>"
    )


def _dossier_evidence_row(item: dict[str, Any]) -> str:
    status = "hashed" if item.get("exists") and item.get("sha256") else "missing"
    return (
        "<tr>"
        f"<td><code>{_h(item.get('path'))}</code></td>"
        f"<td><span class=\"status {'ok' if status == 'hashed' else 'blocked'}\">{status}</span></td>"
        f"<td>{_h(item.get('size_bytes') or 0)}</td>"
        f"<td><code>{_h(item.get('sha256'))}</code></td>"
        "</tr>"
    )


def _render_customer_project_proposal_bundle_html(proposal: dict[str, Any]) -> str:
    customer = _mapping(proposal.get("customer"))
    site = _mapping(proposal.get("site"))
    package = _mapping(proposal.get("customer_project_package"))
    dossier = _mapping(proposal.get("acceptance_dossier"))
    launch_readiness = _mapping(proposal.get("launch_readiness"))
    proposal_insert = _mapping(proposal.get("proposal_insert"))
    readable_delivery = _mapping(proposal.get("customer_readable_delivery"))
    delivery_chain = _mapping(readable_delivery.get("delivery_chain") or proposal.get("delivery_chain"))
    applicability = _mapping(readable_delivery.get("applicability_scope"))
    prerequisites = (
        readable_delivery.get("customer_prerequisites")
        if isinstance(readable_delivery.get("customer_prerequisites"), list)
        else []
    )
    scenario_criteria = (
        readable_delivery.get("scenario_acceptance_criteria")
        if isinstance(readable_delivery.get("scenario_acceptance_criteria"), list)
        else []
    )
    dependency_matrix = (
        readable_delivery.get("dependency_matrix")
        if isinstance(readable_delivery.get("dependency_matrix"), list)
        else []
    )
    delivery_chain_steps = (
        delivery_chain.get("steps")
        if isinstance(delivery_chain.get("steps"), list)
        else []
    )
    release_bundle = _mapping(proposal.get("approved_template_release_bundle"))
    release_notes = release_bundle.get("release_notes") if isinstance(release_bundle.get("release_notes"), list) else []
    gates = dossier.get("gates") if isinstance(dossier.get("gates"), list) else []
    launch_gates = launch_readiness.get("gates") if isinstance(launch_readiness.get("gates"), list) else []
    safe_claims = proposal_insert.get("safe_claims") if isinstance(proposal_insert.get("safe_claims"), list) else []
    boundaries = (
        proposal_insert.get("delivery_boundaries")
        if isinstance(proposal_insert.get("delivery_boundaries"), list)
        else []
    )
    note_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(item).get('template_id'))}</td>"
        f"<td>{_h(_mapping(item).get('version'))}</td>"
        f"<td>{_h(_mapping(item).get('product_status'))}</td>"
        f"<td>{_h(_mapping(item).get('customer_status') or _mapping(item).get('customer_claim'))}</td>"
        "</tr>"
        for item in release_notes
    ) or "<tr><td colspan=\"4\">No approved published template releases are available.</td></tr>"
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(gate).get('label') or _mapping(gate).get('gate_id'))}</td>"
        f"<td>{_h(_mapping(gate).get('status'))}</td>"
        f"<td>{_h(_mapping(gate).get('next_step'))}</td>"
        "</tr>"
        for gate in gates
    ) or "<tr><td colspan=\"3\">No acceptance gates recorded.</td></tr>"
    launch_gate_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(gate).get('label') or _mapping(gate).get('gate_id'))}</td>"
        f"<td>{_h(_mapping(gate).get('status'))}</td>"
        f"<td>{_h(_mapping(gate).get('next_step'))}</td>"
        "</tr>"
        for gate in launch_gates
        if isinstance(gate, dict)
    ) or "<tr><td colspan=\"3\">No launch readiness gates recorded.</td></tr>"
    claim_rows = "".join(f"<li>{_h(item)}</li>" for item in safe_claims)
    boundary_rows = "".join(f"<li>{_h(item)}</li>" for item in boundaries)
    prerequisite_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(item).get('label') or _mapping(item).get('prerequisite_id'))}</td>"
        f"<td>{_h(_mapping(item).get('status'))}</td>"
        f"<td>{_h(_mapping(item).get('owner'))}</td>"
        f"<td>{_h(_mapping(item).get('next_step'))}</td>"
        "</tr>"
        for item in prerequisites
        if isinstance(item, dict)
    ) or "<tr><td colspan=\"4\">No customer prerequisites recorded.</td></tr>"
    scenario_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(item).get('scenario_id'))}</td>"
        f"<td>{_h(', '.join(_string_list(_mapping(item).get('managed_object_labels'))))}</td>"
        f"<td>{_h(', '.join(_string_list(_mapping(item).get('required_evidence'))))}</td>"
        f"<td>{_h(_mapping(item).get('pass_condition'))}</td>"
        "</tr>"
        for item in scenario_criteria
        if isinstance(item, dict)
    ) or "<tr><td colspan=\"4\">No scenario acceptance criteria recorded.</td></tr>"
    delivery_chain_rows = "\n".join(
        "<tr>"
        f"<td>{_h(_mapping(item).get('label') or _mapping(item).get('step_id'))}</td>"
        f"<td>{_h(_mapping(item).get('status'))}</td>"
        f"<td>{_h(_mapping(item).get('evidence'))}</td>"
        f"<td>{_h(_mapping(item).get('next_step'))}</td>"
        "</tr>"
        for item in delivery_chain_steps
        if isinstance(item, dict)
    ) or "<tr><td colspan=\"4\">No customer delivery chain recorded.</td></tr>"
    package_manifest = _mapping(package.get("manifest"))
    dossier_manifest = _mapping(dossier.get("manifest"))
    proposal_manifest = _mapping(proposal.get("manifest"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>AskMe 客户项目提案包</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #17211f; margin: 32px; }}
    header {{ border-bottom: 1px solid #d9e5df; padding-bottom: 18px; margin-bottom: 22px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    .muted {{ color: #64746e; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d9e5df; border-radius: 10px; padding: 14px; }}
    .metric b {{ display: block; font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid #e7efeb; text-align: left; padding: 9px 8px; vertical-align: top; }}
    th {{ color: #42534d; font-size: 12px; text-transform: uppercase; }}
    .boundary {{ margin-top: 18px; padding: 14px; background: #f4faf7; border: 1px solid #cfe4da; border-radius: 10px; }}
    code {{ background: #eef4f1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>AskMe 客户项目提案包</h1>
    <div class="muted">{_h(customer.get("customer_name") or customer.get("customer_id"))} / {_h(customer.get("project_name") or customer.get("project_id"))}</div>
    <div class="muted">{_h(site.get("name") or site.get("site_name") or site.get("site_id"))}</div>
  </header>
  <section class="metrics">
    <div class="metric"><b>{_h(package_manifest.get("managed_object_count"))}</b><span>现场对象</span></div>
    <div class="metric"><b>{_h(package_manifest.get("acceptance_overall_status"))}</b><span>交付包验收</span></div>
    <div class="metric"><b>{_h(dossier.get("overall_status"))}</b><span>验收资料状态</span></div>
    <div class="metric"><b>{_h(launch_readiness.get("launch_stage"))}</b><span>上线阶段</span></div>
  </section>
  <section class="boundary">
    <strong>上线准入</strong>
    <p>{_h(launch_readiness.get("customer_status"))}</p>
    <p>{_h(launch_readiness.get("release_claim"))}</p>
    <table>
      <thead><tr><th>门禁</th><th>状态</th><th>下一步</th></tr></thead>
      <tbody>{launch_gate_rows}</tbody>
    </table>
  </section>
  <section class="boundary">
    <strong>{_h(proposal_insert.get("section_title") or "已发布可复用能力")}</strong>
    <p>{_h(proposal_insert.get("customer_message"))}</p>
    <ul>{claim_rows}</ul>
  </section>
  <h2>客户交付范围</h2>
  <section class="metrics">
    <div class="metric"><b>{_h(', '.join(_string_list(applicability.get("industries"))) or '-')}</b><span>适用行业</span></div>
    <div class="metric"><b>{_h(len(_string_list(applicability.get("scenarios"))))}</b><span>场景数量</span></div>
    <div class="metric"><b>{_h(len(dependency_matrix))}</b><span>依赖项</span></div>
    <div class="metric"><b>{_h(len(prerequisites))}</b><span>客户准备项</span></div>
  </section>
  <table>
    <thead><tr><th>准备项</th><th>状态</th><th>负责人</th><th>下一步</th></tr></thead>
    <tbody>{prerequisite_rows}</tbody>
  </table>
  <table>
    <thead><tr><th>场景</th><th>现场对象</th><th>证据</th><th>通过条件</th></tr></thead>
    <tbody>{scenario_rows}</tbody>
  </table>
  <h2>客户交付链路</h2>
  <table>
    <thead><tr><th>步骤</th><th>状态</th><th>证据</th><th>下一步</th></tr></thead>
    <tbody>{delivery_chain_rows}</tbody>
  </table>
  <h2>已发布模板说明</h2>
  <table>
    <thead><tr><th>模板</th><th>版本</th><th>状态</th><th>客户状态</th></tr></thead>
    <tbody>{note_rows}</tbody>
  </table>
  <h2>验收门禁</h2>
  <table>
    <thead><tr><th>门禁</th><th>状态</th><th>下一步</th></tr></thead>
    <tbody>{gate_rows}</tbody>
  </table>
  <section class="boundary">
    <strong>交付边界</strong>
    <p>{_h(proposal.get("delivery_boundary"))}</p>
    <ul>{boundary_rows}</ul>
    <p class="muted">Proposal SHA-256: <code>{_h(proposal_manifest.get("payload_sha256"))}</code></p>
    <p class="muted">Package SHA-256: <code>{_h(package_manifest.get("payload_sha256"))}</code></p>
    <p class="muted">Dossier SHA-256: <code>{_h(dossier_manifest.get("payload_sha256"))}</code></p>
  </section>
</body>
</html>
"""


def _metric(label: str, value: Any) -> str:
    return (
        "<div class=\"metric\">"
        f"<div class=\"label\">{_h(label)}</div>"
        f"<div class=\"value\">{_h(value if value not in (None, '') else '-')}</div>"
        "</div>"
    )


def _status_class(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"ready", "production_ready"}:
        return "ok"
    if text in {"manual_check", "ready_for_lab", "ready_for_onsite_acceptance"}:
        return "manual_check"
    if text in {"blocked", "failed", "missing", "invalid"}:
        return "blocked"
    return "warn"


def _h(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)

__all__ = [
    "_dossier_evidence_row",
    "_dossier_gate_row",
    "_dossier_workflow_row",
    "_h",
    "_metric",
    "_render_customer_project_acceptance_dossier_html",
    "_render_customer_project_proposal_bundle_html",
    "_status_class",
]
