# AI Research Paper Presentation Checklist

A comprehensive checklist for common presentation-level mistakes and regulations in AI/ML/Security research papers.

---

## 1. Figures and Tables

### Captions
- [ ] Every figure/table has a caption
- [ ] Captions end with a period (full stop)
- [ ] Captions are self-contained (reader can understand without reading main text)
- [ ] Captions describe: what is shown, setup/conditions, key takeaway
- [ ] Caption text is consistent in style (e.g., all start with verbs or all are noun phrases)

### Labels and References
- [ ] Every figure/table has a `\label{}` immediately after `\caption{}`
- [ ] All figures/tables are referenced in the text using `\ref{}` or `\autoref{}`
- [ ] Labels follow consistent naming convention (e.g., `fig:`, `tab:`, `sec:`, `eq:`)
- [ ] No orphan figures/tables (defined but never referenced)
- [ ] References appear *before* the figure/table in reading order (forward reference is acceptable but backward is preferred)

### Formatting
- [ ] Figures are vector graphics (PDF) when possible, not rasterized (PNG/JPG)
- [ ] Figure text is legible at print size (minimum ~8pt font)
- [ ] Axes are labeled with units
- [ ] Legend entries are meaningful (not just "Series 1")
- [ ] Color choices are colorblind-friendly
- [ ] Figures fit within column/page margins
- [ ] Tables use `\toprule`, `\midrule`, `\bottomrule` (booktabs) instead of `\hline`
- [ ] No vertical lines in tables (generally discouraged in academic writing)
- [ ] Consistent decimal precision across table columns
- [ ] Best results are highlighted consistently (bold, underline)

---

## 2. Equations and Math

### Labels and References
- [ ] Important equations have `\label{}` and are referenced with `\eqref{}` or `(\ref{})`
- [ ] Equation numbers are only for equations that are referenced
- [ ] Inline math uses `$...$`, display math uses `\[...\]` or `equation` environment

### Notation Consistency
- [ ] Notation is defined before first use
- [ ] Same symbol means the same thing throughout the paper
- [ ] Vectors are consistently formatted (bold, arrow, or underline)
- [ ] Matrices are consistently formatted (bold uppercase, calligraphic)
- [ ] Sets use consistent notation (calligraphic, blackboard bold)
- [ ] Subscripts/superscripts are consistent (e.g., always `x_i` not sometimes `x^i`)

### Formatting
- [ ] Operators use `\operatorname{}` or predefined commands (`\log`, `\max`, `\arg\min`)
- [ ] Function names are not italicized (use `\text{}` or `\mathrm{}`)
- [ ] Parentheses scale with content using `\left(` and `\right)`
- [ ] No orphan equation numbers (numbered but never referenced)
- [ ] Punctuation after display equations when grammatically required

---

## 3. Citations and References

### Citation Style
- [ ] Citations use consistent format (`\cite{}`, `\citep{}`, `\citet{}` as appropriate)
- [ ] Multiple citations are grouped: `\cite{a,b,c}` not `\cite{a}\cite{b}\cite{c}`
- [ ] Citations are inside the sentence punctuation: "...as shown previously~\cite{foo}."
- [ ] Use `~` (non-breaking space) before `\cite{}` to prevent line breaks

### Bibliography Quality
- [ ] All referenced works are in the bibliography
- [ ] No orphan bibliography entries (listed but never cited)
- [ ] Author names are consistent (same author spelled the same way)
- [ ] Venues are complete (conference name, year, pages)
- [ ] arXiv papers include arXiv ID
- [ ] Published versions preferred over arXiv when available
- [ ] URLs are included for software/datasets
- [ ] DOIs included when available

### Citation Content
- [ ] Claims are supported by citations
- [ ] Citations actually support the claim made (not tangentially related)
- [ ] Recent relevant work is cited (within last 2-3 years)
- [ ] Foundational/seminal works are cited
- [ ] No excessive self-citation

---

## 4. Text and Writing

### Grammar and Style
- [ ] Consistent tense (present for general truths, past for experiments)
- [ ] Consistent voice (active vs. passive, though active preferred)
- [ ] No dangling modifiers
- [ ] No run-on sentences
- [ ] No sentence fragments
- [ ] Acronyms defined on first use: "Self-Supervised Learning (SSL)"
- [ ] Acronyms used consistently after definition
- [ ] Technical terms defined on first use

### LaTeX-Specific
- [ ] Use `--` for en-dash (ranges), `---` for em-dash
- [ ] Use `~` for non-breaking spaces (before citations, after "Fig.", etc.)
- [ ] Quotation marks use ``` `` ``` and `''` not `"`
- [ ] No manual line breaks (`\\`) in paragraphs
- [ ] No manual page breaks unless absolutely necessary
- [ ] `\%` for percent sign, `\&` for ampersand
- [ ] Proper spacing after periods: `\ ` after abbreviations (e.g., "e.g.\ ")

### Numbers and Units
- [ ] Consistent number formatting (e.g., always "1,000" or always "1000")
- [ ] Units have space before them: "10 GB" not "10GB"
- [ ] Use `\times` for multiplication, not `x`
- [ ] Percentage: consistent "50%" or "50 percent"
- [ ] Decimal points, not commas, for decimals (in English)

---

## 5. Structure and Organization

### Sections
- [ ] All sections have `\label{}` and can be referenced
- [ ] Section numbering is correct (no skipped numbers)
- [ ] Section titles are parallel in structure
- [ ] Sections are referenced as "Section~\ref{}" with capital S
- [ ] No single subsection (if you have 4.1, you need 4.2)

### Abstract
- [ ] States the problem
- [ ] States the approach
- [ ] States key results with numbers
- [ ] States the contribution/impact
- [ ] No citations in abstract (some venues prohibit)
- [ ] No undefined acronyms in abstract
- [ ] Within word limit

### Introduction
- [ ] Clearly states the problem and motivation
- [ ] Identifies the gap in prior work
- [ ] Presents the proposed solution
- [ ] Lists contributions (typically 3-5)
- [ ] Provides paper outline (optional but common)

---

## 6. Anonymity (Double-Blind Venues)

- [ ] No author names in paper
- [ ] No identifying information in acknowledgments
- [ ] No "our previous work" with identifiable citations
- [ ] No institutional references ("our university cluster")
- [ ] No identifiable file paths in code snippets
- [ ] Supplementary material is also anonymized
- [ ] GitHub/code links are anonymized (e.g., anonymous.4open.science)
- [ ] No author names in PDF metadata
- [ ] Figure filenames don't contain identifying information

---

## 7. Reproducibility

- [ ] Datasets named with citations/URLs
- [ ] Model architectures specified (or cited)
- [ ] Hyperparameters listed (learning rate, batch size, epochs, etc.)
- [ ] Random seeds mentioned (even if not fixed, state this)
- [ ] Hardware specified (GPU type, memory)
- [ ] Training time reported
- [ ] Number of runs/trials reported
- [ ] Error bars or confidence intervals where applicable
- [ ] Code availability statement

---

## 8. Common LaTeX Errors

### Compilation
- [ ] No undefined references (`??` in output)
- [ ] No overfull/underfull hbox warnings (or minimized)
- [ ] No missing `$` for math mode
- [ ] No unescaped special characters (`_`, `%`, `&`, `#`)
- [ ] All `\begin{}` have matching `\end{}`
- [ ] All `{` have matching `}`

### Floats (Figures/Tables)
- [ ] Floats don't pile up at end of document
- [ ] Use `[t]`, `[b]`, `[h]`, `[H]` appropriately
- [ ] `\caption{}` comes *before* `\label{}`
- [ ] Wide figures use `figure*` in two-column format

### Cross-References
- [ ] Run LaTeX twice (or use latexmk) to resolve all references
- [ ] No circular references
- [ ] All `\ref{}` point to existing `\label{}`

---

## 9. Venue-Specific Requirements

### General
- [ ] Page limit respected (check if references count)
- [ ] Correct template/style file used
- [ ] Font size requirements met
- [ ] Margin requirements met
- [ ] Line numbering (if required for submission)

### USENIX Security Specific
- [ ] Threat model clearly specified
- [ ] Attacker capabilities defined
- [ ] Defender assumptions stated
- [ ] Limitations discussed
- [ ] Ethical considerations addressed (if applicable)
- [ ] Artifact appendix (if submitting artifacts)

---

## 10. Final Checks Before Submission

- [ ] Spell check completed
- [ ] Grammar check completed (Grammarly, LanguageTool, etc.)
- [ ] All TODO/FIXME comments removed
- [ ] All author comment macros removed or hidden (`\nhat{}`, `\mn{}`, etc.)
- [ ] PDF is correct size and readable
- [ ] All fonts are embedded in PDF
- [ ] Supplementary material is included and referenced
- [ ] Co-authors have reviewed and approved
- [ ] Paper compiles without errors on a clean system
- [ ] Submission system accepts the PDF

---

## Quick Reference: Common Mistakes

| Mistake | Wrong | Correct |
|---------|-------|---------|
| Missing period in caption | `\caption{Results}` | `\caption{Results.}` |
| Citation outside punctuation | `shown previously. \cite{foo}` | `shown previously~\cite{foo}.` |
| Undefined acronym | `We use SSL...` | `We use Self-Supervised Learning (SSL)...` |
| Math operator italicized | `$log(x)$` | `$\log(x)$` |
| Hard-coded numbers | `Figure 3 shows...` | `Figure~\ref{fig:results} shows...` |
| Missing non-breaking space | `Figure \ref{fig:a}` | `Figure~\ref{fig:a}` |
| Wrong quotation marks | `"quoted text"` | ``` ``quoted text'' ``` |
| En-dash for ranges | `10-20` | `10--20` |
| No space before units | `10GB` | `10~GB` or `10 GB` |
| Orphan reference | `Table ?? shows...` | Run LaTeX again |

---

## Recommended Tools

- **Spell/Grammar**: Grammarly, LanguageTool, aspell
- **LaTeX Linting**: chktex, lacheck
- **Bibliography**: JabRef, Zotero with Better BibTeX
- **PDF Check**: pdffonts (check embedded fonts)
- **Compilation**: latexmk (handles multiple passes automatically)
- **Version Control**: Git (track all changes)
- **Collaboration**: Overleaf (real-time collaboration)

---

*Last updated: January 2026*
