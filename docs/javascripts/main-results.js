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
        const headMarkup = Array.from({ length: 4 }, (_, headIndex) => `
          <article
            class="architecture-head architecture-head-${headIndex}"
            aria-label="Attention head ${headIndex} computation"
          >
            <header class="architecture-head-header">
              <span class="architecture-head-index">H${headIndex}</span>
              <div>
                <strong>Attention head ${headIndex}</strong>
                <small>independent learned weights</small>
              </div>
            </header>
            <div class="architecture-projections">
              <div>
                <b>Q</b>
                <span>Q<sub>${headIndex}</sub> = R W<sub>Q</sub><sup>${headIndex}</sup></span>
                <small>W<sub>Q</sub><sup>${headIndex}</sup>: 64 &times; 16</small>
              </div>
              <div>
                <b>K</b>
                <span>K<sub>${headIndex}</sub> = R W<sub>K</sub><sup>${headIndex}</sup></span>
                <small>W<sub>K</sub><sup>${headIndex}</sup>: 64 &times; 16</small>
              </div>
              <div>
                <b>Value sources</b>
                <span>S<sub>${headIndex}</sub> = R W<sub>V</sub><sup>${headIndex}</sup></span>
                <small>W<sub>V</sub><sup>${headIndex}</sup>: 64 &times; 16</small>
              </div>
            </div>
            <div class="architecture-attention-equation">
              <span class="architecture-step-label">Full attention matrix</span>
              <strong>
                A<sub>${headIndex}</sub> = softmax((Q<sub>${headIndex}</sub>K<sub>${headIndex}</sub><sup>T</sup>) / &radic;16 + M<sub>causal</sub>)
              </strong>
              <small>N &times; N</small>
            </div>
            <div class="architecture-head-path">
              <div class="architecture-head-step">
                <span class="architecture-step-label">ANS attention row</span>
                <strong>a<sub>${headIndex}</sub> = A<sub>${headIndex}</sub>[-1, :]</strong>
                <small>1 &times; N</small>
              </div>
              <span class="architecture-down-arrow" aria-hidden="true">&#8595;</span>
              <div class="architecture-head-step">
                <span class="architecture-step-label">Weighted value vector</span>
                <strong>V<sub>${headIndex}</sub> = a<sub>${headIndex}</sub>S<sub>${headIndex}</sub></strong>
                <small>1 &times; 16</small>
              </div>
              <span class="architecture-down-arrow" aria-hidden="true">&#8595;</span>
              <div class="architecture-head-step architecture-head-output">
                <span class="architecture-step-label">Residual-stream write</span>
                <strong>z<sub>${headIndex}</sub> = V<sub>${headIndex}</sub>W<sub>O</sub><sup>${headIndex}</sup></strong>
                <small>W<sub>O</sub><sup>${headIndex}</sup>: 16 &times; 64 &nbsp;&rarr;&nbsp; z<sub>${headIndex}</sub>: 1 &times; 64</small>
              </div>
            </div>
          </article>
        `).join("");

        root.innerHTML = `
          <div class="architecture-rules" aria-label="Model rules">
            <div><strong>1</strong><span>attention-only layer</span></div>
            <div><strong>64</strong><span>residual dimensions</span></div>
            <div><strong>4 &times; 16</strong><span>heads &times; dimensions</span></div>
            <div><strong>Causal</strong><span>masked self-attention</span></div>
            <div><strong>[ANS]</strong><span>final row is read out</span></div>
          </div>
          <div class="architecture-source-flow">
            <div class="architecture-source architecture-source-tokens">
              <span class="architecture-eyebrow">Input sequence</span>
              <code>[BOS] n0 [SEP] n1 ... n4 [ANS]</code>
              <small>N tokens</small>
            </div>
            <span class="architecture-flow-arrow" aria-hidden="true">&#8594;</span>
            <div class="architecture-source architecture-source-residual">
              <span class="architecture-eyebrow">Shared residual stream</span>
              <strong>R = W<sub>E</sub>[token] + P</strong>
              <small>N &times; 64</small>
            </div>
          </div>
          <div class="architecture-fanout">
            <span>R is sent to all four heads in parallel</span>
          </div>
          <div class="architecture-head-grid">
            ${headMarkup}
          </div>
          <div class="architecture-merge-cue">
            <span>Only the four final-position writes continue to the answer readout</span>
          </div>
          <div class="architecture-merge-row">
            <div class="architecture-write-chips" aria-label="Four head output vectors">
              ${Array.from({ length: 4 }, (_, headIndex) => `
                <div class="architecture-write architecture-write-${headIndex}">
                  <strong>z<sub>${headIndex}</sub></strong>
                  <small>1 &times; 64</small>
                </div>
              `).join("")}
            </div>
            <span class="architecture-flow-arrow" aria-hidden="true">&#8594;</span>
            <div class="architecture-sum">
              <span class="architecture-step-label">Summed head write</span>
              <strong>z = &Sigma;<sub>h=0</sub><sup>3</sup> z<sub>h</sub></strong>
              <small>1 &times; 64</small>
            </div>
          </div>
          <div class="architecture-readout">
            <span class="architecture-eyebrow">Final [ANS] computation</span>
            <div class="architecture-readout-flow">
              <div>
                <span class="architecture-step-label">Original residual</span>
                <strong>R[-1, :]</strong>
                <small>1 &times; 64</small>
              </div>
              <b aria-hidden="true">+</b>
              <div>
                <span class="architecture-step-label">Head sum</span>
                <strong>z</strong>
                <small>1 &times; 64</small>
              </div>
              <b aria-hidden="true">=</b>
              <div>
                <span class="architecture-step-label">Final residual</span>
                <strong>R<sub>final</sub>[-1, :]</strong>
                <small>1 &times; 64</small>
              </div>
              <b aria-hidden="true">&#8594;</b>
              <div>
                <span class="architecture-step-label">Unembedding</span>
                <strong>&ell; = R<sub>final</sub>[-1, :] W<sub>U</sub></strong>
                <small>W<sub>U</sub>: 64 &times; 14 &nbsp;&rarr;&nbsp; &ell;: 1 &times; 14</small>
              </div>
              <b aria-hidden="true">&#8594;</b>
              <div class="architecture-prediction">
                <span class="architecture-step-label">Prediction</span>
                <strong>argmax<sub>t</sub> &ell;<sub>t</sub></strong>
                <small>one vocabulary token</small>
              </div>
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
