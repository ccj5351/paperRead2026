# CLAUDE.md — Project Paper Reading 2026

## Overview

Paper reading and study repo. 

---

## Markdown Conventions

### TOC

Always add a TOC at the beginning of every markdown file (right after the `# title`). Keep section titles short and simple so anchors are clean.

### Images

Always use this HTML format for images — never bare `![alt](path)` markdown syntax:

```html
<div align="center">
    <img src="figures/example.png" alt="Short description" width="800">
    <p><em>Figure: Caption text here.</em></p>
</div>
```

---

## Paper Note Template

Each paper entry in a notes file should follow this structure:

```markdown
## N. Paper Title — Venue/Year

**Why / What.** One paragraph: the problem with prior approaches, and what this paper does differently.

**How**  
- Bullet points describing the method, architecture, and key design decisions.
- One bullet per major component or step.

<div align="center">
    <img src="figures/example_arch.png" alt="Short description" width="800">
    <p><em>Figure: Architecture caption.</em></p>
</div>

**Pros and Cons**:   
- **Pros:** 1-2 sentences on the main strengths/advantages over prior work.
- **Cons:** 1-2 sentences on the main limitations or trade-offs.

**Q&As**   
Q1: Question text.

Answer, with supporting figure if relevant.
```

Use `---` as a horizontal rule between consecutive paper entries.

---

## Language

Use **American English** spelling throughout — never British English variants.

### Em Dashes

Do not use em dashes (`—`) anywhere. Use a comma, period, or reword the sentence instead.

Common substitutions:

| British | American |
|---------|----------|
| minimise / optimise / initialise | minimize / optimize / initialize |
| normalise / recognise / analyse | normalize / recognize / analyze |
| colour / behaviour / neighbour | color / behavior / neighbor |
| centre / metre | center / meter |
| cancelled / travelled / labelled | canceled / traveled / labeled |

---

## Github Markdown Syntax

### Alerts
To add an alert, use a special blockquote line specifying the alert type, followed by the alert information in a standard blockquote. Five types of alerts are available:
```markdown
> [!NOTE]
> Useful information that users should know, even when skimming content.

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.
```

### Hiding content with comments
You can tell GitHub to hide content from the rendered Markdown by placing the content in an HTML comment.
```
<!-- This content will not appear in the rendered Markdown -->
```
### Color Text

Standard GitHub Flavored Markdown (GFM) does not natively support text color. While many Markdown processors allow inline HTML like `<span style="color:red">`, GitHub sanitizes this and removes style attributes for security. 

GitHub supports KaTeX for mathematical expressions. You can use the `\color` command within `$` or `$$` delimiters to render colored text.

### LaTeX Math — Subscript Escaping

GitHub's Markdown parser processes `_` as italic markup **before** the math renderer runs. This breaks subscripts in inline math.

**Rule:** In inline `$...$` mode, always escape underscores as `\_`. In display `$$...$$` mode, use plain `_` (no escaping needed).

```markdown
<!-- inline $: use \_ -->
$\mathbf{H}\_{t-1}$   ✓
$\mathbf{H}_{t-1}$    ✗  (renders broken on GitHub)

<!-- display $$: use _ normally -->
$$\mathbf{H}_{t-1}$$  ✓
```

This applies to all subscripts in inline math: `_t`, `_{t-1}`, `_{xi}`, `_i`, etc.

### LaTeX Math — Curly Braces

GitHub's Markdown parser consumes `\` as an escape character **before** the math renderer runs. This means `\{` and `\}` arrive at KaTeX as bare `{` and `}`, which are grouping characters and render invisibly.

**Rule:** Use `\lbrace` and `\rbrace` instead of `\{` and `\}` for visible curly braces in inline math.

```markdown
<!-- inline $: use \lbrace / \rbrace -->
$m \in \lbrace 0, 1 \rbrace^n$   ✓
$m \in \{0, 1\}^n$               ✗  (braces invisible on GitHub)
```

### LaTeX Math — Piecewise / Cases

GitHub's Markdown parser consumes `\\` as an escape sequence **before** the math renderer runs. Inside a single-line `$$...$$` block, the `\\` row separator in `\begin{cases}` is silently dropped, breaking the piecewise layout.

**Rule:** Always write `\begin{cases}` blocks as multi-line display math (opening `$$` on its own line).

```markdown
<!-- each row on its own line: \\ is preserved -->
$$
f(x) = \begin{cases}
1 & x > 0 \\
0 & x \leq 0
\end{cases}
$$

<!-- all on one line: \\ is consumed, cases collapse -->
$$f(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$   ✗
```

### LaTeX Math — Spacing Commands

GitHub's KaTeX renderer does not support certain TeX spacing primitives. The following commands silently produce no output or cause rendering errors:

| Command | Meaning in standard LaTeX | GitHub behavior |
|---|---|---|
| `\;` | thick space (5/18 em) | **not rendered — omit** |
| `\!` | negative thin space (-3/18 em) | **not rendered — omit** |
| `\,` | thin space (3/18 em) | **not rendered — omit** |

**Rule:** Never use `\;`, `\!`, or `\,` in any GitHub Markdown math block. Use a plain space or no space instead.

```markdown
<!-- wrong -->
$\max(|\min\_\text{obs}|,\; |\max\_\text{obs}|)$   ✗
$\text{round}\!\left(\frac{a}{b}\right)$            ✗
$\frac{1}{\sqrt{d}}\, Q$                            ✗

<!-- correct -->
$\max(|\min\_\text{obs}|, |\max\_\text{obs}|)$      ✓
$\text{round}\left(\frac{a}{b}\right)$              ✓
$\frac{1}{\sqrt{d}} Q$                              ✓
```

### LaTeX Math — Unsupported Macros

GitHub's KaTeX allowlist blocks certain standard LaTeX macros. They produce an error rather than silently rendering.

| Macro | Replacement | Example |
|---|---|---|
| `\operatorname{foo}` | `\mathrm{foo}` | `\mathrm{diag}`, `\mathrm{softmax}` |
| `\tag{n}` | `\qquad \text{(n)}` at end of equation | `$$x = \cdots \qquad \text{(1)}$$` |

**Rule:** Never use `\operatorname{}` or `\tag{}`. Use `\mathrm{}` for named operators; use `\qquad \text{(n)}` for equation labels.

```markdown
<!-- wrong -->
$\operatorname{diag}(\mathbf{s})$   ✗  (macro not allowed on GitHub)

<!-- correct -->
$\mathrm{diag}(\mathbf{s})$         ✓
```

### LaTeX Math — Absolute Value Bars in Tables

GitHub's Markdown table parser treats `|` as a column separator **before** the math renderer runs. Bare `|` characters inside inline math in a table cell will break the table.

**Rule:** Inside table cells, always use `\lvert` and `\rvert` instead of `|` for absolute value bars.

```markdown
<!-- wrong — | inside math breaks the table column -->
| Scale | $s_i = \max|X_i|^\alpha$ |

<!-- correct -->
| Scale | $s\_i = \max\lvert X\_i\rvert^\alpha$ |
```

This applies to any `|` that would otherwise appear inside `$...$` within a table row, including absolute values, norms written as `|x|`, and conditional expressions.

### Footnotes
You can add footnotes to your content by using this bracket syntax:
```
Here is a simple footnote[^1].

A footnote can also have multiple lines[^2].

[^1]: My reference.
[^2]: To add line breaks within a footnote, add 2 spaces to the end of a line.  
This is a second line.
```