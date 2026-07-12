(function () {
  "use strict";

  const GROUP_COPY = {
    "max = 0": "All four heads read [ANS]",
    "max = 1": "H3 uses a soft 1 / [ANS] mixture",
    "max = 2-6": "H3 reads the maximum",
    "max = 7-8": "H2 and H3 read the maximum",
    "max = 9": "H0, H2, and H3 read the maximum",
  };

  const COLOR_STOPS = [
    [0.0, [247, 248, 246]],
    [0.20, [217, 238, 231]],
    [0.58, [90, 182, 156]],
    [1.0, [11, 98, 88]],
  ];

  function attentionColor(value) {
    const bounded = Math.max(0, Math.min(1, value));
    let left = COLOR_STOPS[0];
    let right = COLOR_STOPS[COLOR_STOPS.length - 1];
    for (let index = 1; index < COLOR_STOPS.length; index += 1) {
      if (bounded <= COLOR_STOPS[index][0]) {
        left = COLOR_STOPS[index - 1];
        right = COLOR_STOPS[index];
        break;
      }
    }
    const span = right[0] - left[0] || 1;
    const amount = (bounded - left[0]) / span;
    const rgb = left[1].map((channel, index) =>
      Math.round(channel + amount * (right[1][index] - channel)),
    );
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  }

  function tokenName(token) {
    if (token === 10) return "BOS";
    if (token === 11) return "SEP";
    if (token === 12) return "ANS";
    if (token === 13) return "EOS";
    return String(token);
  }

  function shortTokenName(token) {
    if (token === 10) return "B";
    if (token === 11) return "S";
    if (token === 12) return "A";
    if (token === 13) return "E";
    return String(token);
  }

  function element(tagName, className, text) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function initializeArchitecture() {
    document
      .querySelectorAll(".architecture-diagram[data-transformer-architecture]:not([data-rendered])")
      .forEach((root) => {
        root.innerHTML = `
          <div class="architecture-pipeline">
            <div class="architecture-node architecture-node-input">
              <span class="architecture-eyebrow">Input</span>
              <strong>Token sequence</strong>
              <code>[BOS] n0 [SEP] ... n4 [ANS]</code>
              <small>N tokens</small>
            </div>
            <span class="architecture-arrow" aria-hidden="true">&#8594;</span>
            <div class="architecture-node">
              <span class="architecture-eyebrow">Residual stream</span>
              <strong>Embedding + position</strong>
              <span class="architecture-math">R = W<sub>E</sub>[token] + P</span>
              <small>N &times; 64</small>
            </div>
            <span class="architecture-arrow" aria-hidden="true">&#8594;</span>
            <div class="architecture-node architecture-node-attention">
              <span class="architecture-eyebrow">Layer 0</span>
              <strong>Four parallel heads</strong>
              <span>H0 &nbsp; H1 &nbsp; H2 &nbsp; H3</span>
              <small>16 dimensions per head</small>
            </div>
            <span class="architecture-arrow" aria-hidden="true">&#8594;</span>
            <div class="architecture-node">
              <span class="architecture-eyebrow">Readout</span>
              <strong>Unembedding</strong>
              <span class="architecture-math">logits = R<sub>final</sub> W<sub>U</sub></span>
              <small>N &times; 14</small>
            </div>
            <span class="architecture-arrow" aria-hidden="true">&#8594;</span>
            <div class="architecture-node architecture-node-output">
              <span class="architecture-eyebrow">Prediction</span>
              <strong>[ANS] row</strong>
              <span>argmax over vocabulary</span>
              <small>1 &times; 14</small>
            </div>
          </div>
          <div class="architecture-head-detail">
            <div class="architecture-head-label">
              <span class="architecture-eyebrow">Inside head h</span>
              <strong>ANS selects a source value, then writes to the residual stream</strong>
            </div>
            <div class="architecture-qkv">
              <div><strong>Query</strong><span>Q<sub>h</sub> = R W<sub>Q</sub><sup>h</sup></span><small>N &times; 16</small></div>
              <div><strong>Key</strong><span>K<sub>h</sub> = R W<sub>K</sub><sup>h</sup></span><small>N &times; 16</small></div>
              <div><strong>Value</strong><span>V<sub>source,h</sub> = R W<sub>V</sub><sup>h</sup></span><small>N &times; 16</small></div>
            </div>
            <div class="architecture-head-flow">
              <span>Q<sub>h</sub>K<sub>h</sub><sup>T</sup></span><b aria-hidden="true">&#8594;</b>
              <span>softmax attention A<sub>h</sub></span><b aria-hidden="true">&#8594;</b>
              <span>A<sub>h</sub>V<sub>source,h</sub></span><b aria-hidden="true">&#8594;</b>
              <span>W<sub>O</sub><sup>h</sup></span><b aria-hidden="true">&#8594;</b>
              <span>head write, N &times; 64</span>
            </div>
            <div class="architecture-residual-line">
              <span>sum H0 + H1 + H2 + H3</span><b aria-hidden="true">+</b>
              <span>residual R</span><b aria-hidden="true">&#8594;</b><span>R<sub>final</sub></span>
            </div>
          </div>
        `;
        root.dataset.rendered = "true";
      });
  }

  function renderCase(caseData) {
    const card = element("article", "attention-case");
    const title = element("div", "attention-case-title", `max = ${caseData.max_value}`);
    card.appendChild(title);

    const matrix = element("div", "attention-matrix");
    matrix.setAttribute("role", "table");
    matrix.setAttribute(
      "aria-label",
      `ANS attention by head when the maximum is ${caseData.max_value}`,
    );

    const corner = element("div", "attention-corner", "head");
    corner.setAttribute("aria-hidden", "true");
    matrix.appendChild(corner);

    caseData.tokens.forEach((token, position) => {
      const label = element("div", "attention-source-label");
      label.innerHTML = `
        <span class="attention-token-full">${tokenName(token)}</span>
        <span class="attention-token-short">${shortTokenName(token)}</span>
        <small>${position}</small>
      `;
      label.title = `${tokenName(token)} at source position ${position}`;
      matrix.appendChild(label);
    });

    caseData.attention.forEach((row, headIndex) => {
      const head = element("div", "attention-head-label", `H${headIndex}`);
      head.setAttribute("role", "rowheader");
      matrix.appendChild(head);
      row.forEach((probability, position) => {
        const cell = element("div", "attention-cell");
        cell.style.backgroundColor = attentionColor(probability);
        cell.title = `H${headIndex} to ${tokenName(caseData.tokens[position])}@${position}: ${(
          probability * 100
        ).toFixed(3)}%`;
        cell.setAttribute("role", "cell");
        cell.setAttribute("aria-label", cell.title);
        if (position === caseData.top_positions_by_head[headIndex]) {
          cell.classList.add("is-row-maximum");
        }
        matrix.appendChild(cell);
      });
    });

    card.appendChild(matrix);
    return card;
  }

  function renderAttention(root, data) {
    const cases = new Map(
      data.individual_cases.map((caseData) => [caseData.max_value, caseData]),
    );
    const fragment = document.createDocumentFragment();

    data.groups.forEach((group, groupIndex) => {
      const section = element("section", "attention-regime");
      section.dataset.count = String(group.max_values.length);

      const header = element("header", "attention-regime-header");
      header.appendChild(
        element("span", "attention-regime-number", String(groupIndex + 1).padStart(2, "0")),
      );
      const headerCopy = element("div", "attention-regime-copy");
      headerCopy.appendChild(element("h4", "attention-regime-title", group.label));
      headerCopy.appendChild(
        element("p", "attention-regime-summary", GROUP_COPY[group.label] || ""),
      );
      header.appendChild(headerCopy);
      section.appendChild(header);

      const caseRow = element("div", "attention-case-row");
      group.max_values.forEach((maxValue) => {
        caseRow.appendChild(renderCase(cases.get(maxValue)));
      });
      section.appendChild(caseRow);
      fragment.appendChild(section);
    });

    const scale = element("div", "attention-scale");
    scale.setAttribute("role", "img");
    scale.setAttribute("aria-label", "Shared attention probability scale from zero to one hundred percent");
    scale.innerHTML = `
      <div class="attention-scale-bar"></div>
      <div class="attention-scale-labels">
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
      <div class="attention-scale-caption">Attention probability</div>
    `;
    fragment.appendChild(scale);

    root.replaceChildren(fragment);
    root.dataset.rendered = "true";
  }

  async function initializeAttentionGrids() {
    const roots = document.querySelectorAll(
      ".attention-regime-grid[data-attention-source]:not([data-rendered])",
    );
    await Promise.all(
      Array.from(roots).map(async (root) => {
        try {
          const response = await fetch(root.dataset.attentionSource);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          renderAttention(root, await response.json());
        } catch (error) {
          root.replaceChildren(
            element(
              "p",
              "attention-error",
              "The attention matrices could not be loaded. Open the exact values below.",
            ),
          );
        }
      }),
    );
  }

  function initializeMainResults() {
    const marker = document.querySelector("[data-main-results-page]");
    const article = marker && marker.closest(".md-content__inner");
    if (!article) return Promise.resolve();
    article.classList.add("main-results");
    initializeArchitecture();
    return initializeAttentionGrids();
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(initializeMainResults);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeMainResults);
  } else {
    initializeMainResults();
  }
})();
