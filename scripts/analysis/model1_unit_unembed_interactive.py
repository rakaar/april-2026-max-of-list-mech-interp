#!/usr/bin/env python3
"""Build an interactive 3D angular-geometry explorer for unit-W_U retrains."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.offline
import torch

from model1_unit_unembed_experiment import (
    MODEL_LABELS,
    MODEL_ORDER,
    ROOT,
    extract_components,
    fit_pca,
    load_model,
    output_matrices,
    tokenize,
)


JSON_OUT = ROOT / "docs" / "assets" / "model1_unit_unembed_interactive.json"
HTML_OUT = ROOT / "docs" / "assets" / "model1_unit_unembed_interactive.html"
HEAD_COLORS = ("#2563eb", "#f97316", "#16a34a", "#dc2626")
DIGIT_COLORS = (
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#a16207",
    "#db2777",
    "#0891b2",
    "#65a30d",
    "#4f46e5",
)


def matched_numbers(maximum: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[0, 0, maximum, 0, 0]], device=device)


def vector(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu()]


def scores(state: torch.Tensor, candidates: torch.Tensor) -> dict:
    dots = state @ candidates.T
    cosine = torch.nn.functional.cosine_similarity(
        state[None, :], candidates, dim=1
    )
    return {
        "dot": vector(dots),
        "cosine": vector(cosine),
        "dot_prediction": int(dots.argmax()),
        "cosine_prediction": int(cosine.argmax()),
    }


def build_data() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {
        "description": (
            "Actual matched-input geometry for the released model and three unit-row "
            "unembedding retrains. Head writes and final states are projected into each "
            "model's own top-three PCA basis."
        ),
        "models": {},
        "head_colors": list(HEAD_COLORS),
        "digit_colors": list(DIGIT_COLORS),
    }
    for model_name in MODEL_ORDER:
        model, config = load_model(model_name, device)
        unembedding = model.unembed.weight.detach().cpu().double()
        matrices = output_matrices(model).detach().cpu().double()
        pcas = {
            "digit_unembedding": fit_pca(unembedding[:10]),
            "full_unembedding": fit_pca(unembedding),
            "output_matrix": fit_pca(matrices.reshape(64, 64)),
        }
        model_data = {
            "label": MODEL_LABELS[model_name],
            "unit_unembedding": model_name.startswith("unit_"),
            "config": config,
            "bases": {},
        }
        for basis_name, pca in pcas.items():
            basis = pca["basis"][:, :3]
            raw_digits = unembedding[:10] @ basis
            centered_digits = (
                unembedding[:10] - unembedding[:10].mean(dim=0, keepdim=True)
            ) @ basis
            cases = {}
            for maximum in range(10):
                numbers = matched_numbers(maximum, device)
                with torch.no_grad():
                    components = extract_components(model, tokenize(numbers))
                basis_device = basis.to(
                    device=device, dtype=components["final_state"].dtype
                )
                heads = components["head_writes"][0] @ basis_device
                head_sum = components["head_sum"][0] @ basis_device
                ans_residual = components["ans_residual"][0] @ basis_device
                final_state = components["final_state"][0] @ basis_device
                raw_device = raw_digits.to(device=device, dtype=final_state.dtype)
                centered_device = centered_digits.to(
                    device=device, dtype=final_state.dtype
                )
                maximum_candidate_norm = max(
                    float(raw_digits.norm(dim=1).max()),
                    float(centered_digits.norm(dim=1).max()),
                )
                maximum_state_norm = max(
                    float(heads.norm(dim=1).max()),
                    float(head_sum.norm()),
                    float(ans_residual.norm()),
                    float(final_state.norm()),
                )
                display_scale = 0.88 * maximum_candidate_norm / maximum_state_norm
                full_unembedding = model.unembed.weight.detach()[:10]
                full_state = components["final_state"][0]
                full_dots = full_state @ full_unembedding.T
                full_cosines = torch.nn.functional.cosine_similarity(
                    full_state[None, :], full_unembedding, dim=1
                )
                cases[str(maximum)] = {
                    "numbers": [0, 0, maximum, 0, 0],
                    "head_coordinates": [vector(row) for row in heads],
                    "head_sum_coordinates": vector(head_sum),
                    "ans_residual_coordinates": vector(ans_residual),
                    "final_state_coordinates": vector(final_state),
                    "display_scale": display_scale,
                    "raw_scores": scores(final_state, raw_device),
                    "centered_scores": scores(final_state, centered_device),
                    "full64": {
                        "dot_prediction": int(full_dots.argmax()),
                        "cosine_prediction": int(full_cosines.argmax()),
                        "dot": vector(full_dots),
                        "cosine": vector(full_cosines),
                    },
                }
            model_data["bases"][basis_name] = {
                "explained_variance": [
                    float(value) for value in pca["explained"][:3]
                ],
                "top3_cumulative_explained_variance": float(
                    pca["explained"][:3].sum()
                ),
                "raw_digit_coordinates": [vector(row) for row in raw_digits],
                "centered_digit_coordinates": [
                    vector(row) for row in centered_digits
                ],
                "raw_digit_norms": [
                    float(value) for value in raw_digits.norm(dim=1)
                ],
                "centered_digit_norms": [
                    float(value) for value in centered_digits.norm(dim=1)
                ],
                "cases": cases,
            }
        data["models"][model_name] = model_data
    return data


def html_document(data: dict) -> str:
    plotly_js = plotly.offline.get_plotlyjs()
    payload = json.dumps(data, allow_nan=False, separators=(",", ":"))
    model_options = "".join(
        f'<option value="{name}"{" selected" if name == "unit_seed42" else ""}>'
        f'{data["models"][name]["label"]}</option>'
        for name in MODEL_ORDER
    )
    max_options = "".join(
        f'<option value="{maximum}"{" selected" if maximum == 2 else ""}>'
        f'{maximum}</option>'
        for maximum in range(10)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Unit-unembedding angular geometry</title>
  <style>
    html, body {{ margin: 0; background: #fff; color: #111827; font-family: Arial, sans-serif; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: end; padding: 12px 16px 8px; border-bottom: 1px solid #d1d5db; }}
    label {{ display: grid; gap: 4px; color: #475569; font-size: 12px; font-weight: 600; }}
    select {{ min-width: 132px; height: 34px; border: 1px solid #94a3b8; border-radius: 4px; background: #fff; color: #111827; padding: 0 28px 0 8px; font-size: 14px; }}
    #plot {{ width: 100%; height: 820px; }}
    .note {{ padding: 0 16px 12px; color: #64748b; font-size: 12px; }}
    @media (max-width: 720px) {{ #plot {{ height: 980px; }} .toolbar {{ gap: 8px; }} select {{ min-width: 118px; }} }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <div class="toolbar">
    <label>Model<select id="model">{model_options}</select></label>
    <label>PCA basis
      <select id="basis">
        <option value="digit_unembedding">digit W_U</option>
        <option value="full_unembedding">full-vocab W_U</option>
        <option value="output_matrix" selected>stacked W_O</option>
      </select>
    </label>
    <label>Unembedding coordinates
      <select id="coordinate-mode">
        <option value="raw" selected>raw projection</option>
        <option value="centered">centered PCA</option>
      </select>
    </label>
    <label>True maximum<select id="maximum">{max_options}</select></label>
  </div>
  <div id="plot"></div>
  <div class="note">Head, residual, sum, and final-state arrows share one positive display scale. Dot and cosine bars use unscaled three-dimensional coordinates.</div>
  <script>
  const DATA = {payload};
  const DIGIT_COLORS = DATA.digit_colors;
  const HEAD_COLORS = DATA.head_colors;
  const basisLabels = {{digit_unembedding: 'digit W_U', full_unembedding: 'full-vocab W_U', output_matrix: 'stacked W_O'}};

  function lineTrace(name, vector, color, width, dash, symbol, scale, showlegend=true) {{
    const v = vector.map(x => x * scale);
    return {{type:'scatter3d', mode:'lines+markers+text', scene:'scene', name, showlegend,
      x:[0,v[0]], y:[0,v[1]], z:[0,v[2]],
      line:{{color,width,dash}}, marker:{{color,size:[2,7],symbol}},
      text:[null,name], textposition:'top center', textfont:{{color,size:11}},
      hovertemplate:`<b>${{name}}</b><br>raw 3D (${{vector.map(x => x.toFixed(4)).join(', ')}})<br>display scale ${{scale.toPrecision(4)}}<extra></extra>`}};
  }}

  function render() {{
    const modelName = document.getElementById('model').value;
    const basisName = document.getElementById('basis').value;
    const mode = document.getElementById('coordinate-mode').value;
    const maximum = document.getElementById('maximum').value;
    const model = DATA.models[modelName];
    const basis = model.bases[basisName];
    const item = basis.cases[maximum];
    const U = mode === 'raw' ? basis.raw_digit_coordinates : basis.centered_digit_coordinates;
    const score = mode === 'raw' ? item.raw_scores : item.centered_scores;
    const traces = [];
    for (let digit=0; digit<10; digit++) {{
      const u = U[digit];
      traces.push({{type:'scatter3d', mode:'lines+markers+text', scene:'scene', name:`U${{digit}}`, showlegend:false,
        x:[0,u[0]], y:[0,u[1]], z:[0,u[2]], line:{{color:DIGIT_COLORS[digit],width:3}},
        marker:{{color:DIGIT_COLORS[digit],size:[1,6]}}, text:[null,`U${{digit}}`], textposition:'top center',
        hovertemplate:`<b>U${{digit}}</b><br>(${{u.map(x => x.toFixed(5)).join(', ')}})<br>norm ${{Math.hypot(...u).toFixed(5)}}<extra></extra>`}});
    }}
    for (let head=0; head<4; head++) traces.push(lineTrace(`H${{head}}`, item.head_coordinates[head], HEAD_COLORS[head], 6, 'solid', 'circle', item.display_scale));
    traces.push(lineTrace('[ANS] residual', item.ans_residual_coordinates, '#64748b', 4, 'dot', 'square', item.display_scale));
    traces.push(lineTrace('head sum', item.head_sum_coordinates, '#111827', 6, 'dash', 'diamond', item.display_scale));
    traces.push(lineTrace('final state', item.final_state_coordinates, '#000000', 10, 'solid', 'diamond', item.display_scale));
    const barColors = Array.from({{length:10}}, (_,d) => d === Number(maximum) ? '#0f766e' : '#cbd5e1');
    traces.push({{type:'bar', x:[0,1,2,3,4,5,6,7,8,9], y:score.dot, xaxis:'x', yaxis:'y', name:'dot', showlegend:false,
      marker:{{color:barColors}}, text:score.dot.map(x=>x.toFixed(2)), textposition:'outside', cliponaxis:false,
      hovertemplate:'digit %{{x}}<br>dot %{{y:.6f}}<extra></extra>'}});
    traces.push({{type:'bar', x:[0,1,2,3,4,5,6,7,8,9], y:score.cosine, xaxis:'x2', yaxis:'y2', name:'cosine', showlegend:false,
      marker:{{color:barColors}}, text:score.cosine.map(x=>x.toFixed(3)), textposition:'outside', cliponaxis:false,
      hovertemplate:'digit %{{x}}<br>cosine %{{y:.6f}}<extra></extra>'}});
    const allPoints = U.concat(item.head_coordinates.map(v=>v.map(x=>x*item.display_scale)), [item.head_sum_coordinates.map(x=>x*item.display_scale), item.final_state_coordinates.map(x=>x*item.display_scale)]);
    const extent = Math.max(0.2, ...allPoints.flat().map(Math.abs)) * 1.18;
    const title = `<b>${{model.label}} | ${{basisLabels[basisName]}} top 3 | ${{mode}} U | true max ${{maximum}}</b>` +
      `<br><sup>3D dot predicts ${{score.dot_prediction}}; 3D cosine predicts ${{score.cosine_prediction}}; full-64D dot/cosine ${{item.full64.dot_prediction}}/${{item.full64.cosine_prediction}}</sup>`;
    const layout = {{title:{{text:title,x:0.5,xanchor:'center'}}, template:'plotly_white', height:820, margin:{{l:28,r:22,t:92,b:58}}, uirevision:`${{modelName}}-${{basisName}}-${{mode}}`,
      scene:{{domain:{{x:[0,0.67],y:[0,1]}}, aspectmode:'cube', camera:{{eye:{{x:1.45,y:-1.55,z:1.0}}}},
        xaxis:{{title:'PC1',range:[-extent,extent],showspikes:false}}, yaxis:{{title:'PC2',range:[-extent,extent],showspikes:false}}, zaxis:{{title:'PC3',range:[-extent,extent],showspikes:false}}}},
      xaxis:{{domain:[0.74,1],anchor:'y',title:'digit',dtick:1}}, yaxis:{{domain:[0.57,1],anchor:'x',title:`3D dot (winner ${{score.dot_prediction}})`}},
      xaxis2:{{domain:[0.74,1],anchor:'y2',title:'digit',dtick:1}}, yaxis2:{{domain:[0,0.42],anchor:'x2',title:`3D cosine (winner ${{score.cosine_prediction}})`}},
      legend:{{orientation:'h',x:0.32,xanchor:'center',y:1.01,yanchor:'bottom',font:{{size:10}}}}, paper_bgcolor:'#fff', plot_bgcolor:'#fff'}};
    Plotly.react('plot', traces, layout, {{responsive:true,displaylogo:false,scrollZoom:true}});
  }}
  for (const id of ['model','basis','coordinate-mode','maximum']) document.getElementById(id).addEventListener('change', render);
  render();
  </script>
</body>
</html>
"""


def main() -> None:
    torch.manual_seed(0)
    data = build_data()
    JSON_OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    HTML_OUT.write_text(html_document(data))
    for model_name in MODEL_ORDER:
        item = data["models"][model_name]["bases"]["output_matrix"]["cases"]
        print(
            f"{model_name},output_top3_raw_cosine_predictions,"
            + ",".join(str(item[str(maximum)]["raw_scores"]["cosine_prediction"]) for maximum in range(10))
        )
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{HTML_OUT}")


if __name__ == "__main__":
    main()
