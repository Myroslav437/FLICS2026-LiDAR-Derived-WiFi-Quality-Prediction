"""Verify the regenerated unified report against the post-merge file layout.

Pass-3 verification per Stage 2 of the merge spec:
- Figure paths in the unified report resolve to existing files under docs/.
- Required headings (§1..§10 + appendices) are present.
- Canonical numbers per the spec §4.4 are present:
  - σ_intra = 4.95 dB (Hardening D)
  - Δ_LiDAR_placebo on F-B locked = -0.37 dB (Hardening A)
  - Δ_LiDAR_placebo on F-B H1 = +0.01 dB (Hardening A)
  - LightGBM Δ_LiDAR (in-FOV, default, H1-equiv) = -0.58, -0.93 dB (Hardening B)
  - Δ_LiDAR_within for combined H1+buffer = -0.83 dB (Hardening C)
  - Δ_real (+1.11), Δ_placebo (+0.13), gap (0.98) (Hardening E)
  - Cleanest model RMSE = 1.99 dB.
- Eight-experiment, five-orthogonal-axis framing string.
- Nine-row synthesis table.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
REPORT = ROOT / "docs" / "unified_report.md"


def main() -> int:
    text = REPORT.read_text(encoding="utf-8")

    print("=" * 72)
    print(f"Verifying {REPORT.relative_to(ROOT)}")
    print("=" * 72)
    print()

    issues: list[str] = []

    # 1. Figure paths
    print("## Figure path resolution")
    fig_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    figs = [m.group(1) for m in fig_pattern.finditer(text)]
    docs_root = ROOT / "docs"
    n_fig_ok = 0
    n_fig_missing = 0
    for f in figs:
        # Skip external URLs
        if f.startswith(("http://", "https://")):
            continue
        # Markdown image relative to the report's directory
        candidate = docs_root / f
        if candidate.exists():
            n_fig_ok += 1
        else:
            n_fig_missing += 1
            issues.append(f"figure missing: {f}")
            print(f"  MISSING: {f}")
    print(f"  Figures resolved: {n_fig_ok}")
    print(f"  Figures missing: {n_fig_missing}")
    print()

    # 2. Required top-level sections
    print("## Top-level section presence")
    required = [
        "## §1. Executive summary",
        "## §2. Project context and objectives",
        "## §3. Dataset and time synchronization",
        "## §4. Phase 0",
        "## §5. Phase 1 dataset construction",
        "## §6. Project A",
        "## §7. Project B",
        "## §8. Synthesis across all experiments",
        "## §9. Limitations and caveats",
        "## §10. Implications and paper-writing guidance",
        "## Appendix A.",
        "## Appendix B.",
        "## Appendix C.",
        "## Appendix D.",
    ]
    for h in required:
        if h in text:
            print(f"  OK   {h}")
        else:
            print(f"  FAIL {h}")
            issues.append(f"missing section: {h}")
    print()

    # 3. Hardening sections present
    print("## Hardening section presence")
    hardening_sections = [
        ("§3.8", "Dataset noise floor (Hardening D)"),
        ("§6.14", "Hardening A — Cross-session LiDAR placebo on F-B"),
        ("§6.15", "Hardening B — LightGBM cross-check on F-B"),
        ("§7.13.4", "Hardening C — R-4 combined H1 + 1m buffer"),
        ("§7.13.5", "Hardening E — Within-session LiDAR placebo on R-4"),
        ("§7.13.6", "Combined R-4 verdict — four diagnostics"),
    ]
    for ref, title in hardening_sections:
        # Match by partial title in §6.14 / §3.8 / etc.
        if title.split(" — ")[0] in text and ref in text:
            print(f"  OK   {ref} {title}")
        else:
            print(f"  FAIL {ref} {title}")
            issues.append(f"missing hardening section: {ref} {title}")
    print()

    # 4. Canonical numerical consistency
    print("## Canonical numerical consistency")
    canonical = [
        ("σ_intra global", "4.95"),
        ("σ_intra 15.03", "4.901"),
        ("σ_intra 24.03", "5.005"),
        ("same-cell median |Δ|", "4.24 dB"),
        ("Hardening A locked Δ_real", "-0.52"),
        ("Hardening A locked Δ_placebo", "-0.37"),
        ("Hardening A H1 Δ_real", "-0.29"),
        ("Hardening A H1 Δ_placebo", "+0.01"),
        ("Hardening B LightGBM default Δ", "-0.58"),
        ("Hardening B LightGBM H1-equiv Δ", "-0.93"),
        ("Hardening C combined Δ", "-0.83"),
        ("Hardening E Δ_real", "+1.11"),
        ("Hardening E Δ_placebo", "+0.13"),
        ("Hardening E gap", "0.98"),
        ("R-4 main +1.085", "+1.085"),
        ("R-4 buffer -1.267", "-1.267"),
        ("R-4 H1 -0.331", "-0.331"),
        ("R-4 W2 H1 in-FOV", "1.99"),
        ("Phase 1 SHA", "c164c53b5dd272384f32564758b08ad53ec886955ef2e50ce69979e125018270"),
    ]
    n_can_ok = 0
    n_can_miss = 0
    for label, needle in canonical:
        # Use Unicode normalization: search for either Unicode minus or hyphen variants.
        # The needle as written may use ASCII '-'; the report often uses Unicode minus '−'.
        unicode_variant = needle.replace("-", "−")
        if needle in text or unicode_variant in text:
            n_can_ok += 1
            print(f"  OK   {label}: {needle!r}")
        else:
            n_can_miss += 1
            print(f"  MISS {label}: {needle!r} (also tried {unicode_variant!r})")
            issues.append(f"missing canonical: {label}={needle}")
    print(f"  PASS: {n_can_ok}/{len(canonical)}")
    print()

    # 5. Eight-experiment / five-axis framing
    print("## Rev9 framing")
    framing_phrases = [
        "eight robustness checks across five orthogonal axes",
        "nine-experiment, five-orthogonal-axis",
        "Hardening A",
        "Hardening B",
        "Hardening C",
        "Hardening D",
        "Hardening E",
    ]
    for p in framing_phrases:
        n = text.count(p)
        status = "OK" if n > 0 else "MISS"
        print(f"  {status:4s} {p!r}: {n} occurrence(s)")
        if n == 0:
            issues.append(f"missing framing: {p}")
    print()

    # 6. No leftover Rev8 citations in the main body
    print("## No Rev8 citations remaining")
    n_rev8 = text.count("proposal_rev8.md")
    if n_rev8 == 0:
        print("  OK: 0 Rev8 citations")
    else:
        print(f"  FAIL: {n_rev8} leftover Rev8 citations")
        issues.append(f"leftover Rev8: {n_rev8}")
    print()

    # 7. docs/p1_hardening references are allowed only as provenance ("Originally
    #    `docs/p1_hardening/results_report.md`" comments and the §1 sources line that
    #    explains the merge). All 8 expected occurrences live on lines containing
    #    "Originally", "no longer exists", "dissolved from", or "were dissolved".
    print("## docs/p1_hardening references — provenance only")
    provenance_markers = ("Originally", "dissolved", "no longer exists", "were dissolved", "scripts/p1_hardening/")
    bad: list[tuple[int, str]] = []
    n_provenance = 0
    for i, ln in enumerate(text.splitlines(), start=1):
        if "docs/p1_hardening" in ln:
            if any(m in ln for m in provenance_markers):
                n_provenance += 1
            else:
                bad.append((i, ln))
    if not bad:
        print(f"  OK: {n_provenance} provenance annotations; 0 unexpected references")
    else:
        print(f"  FAIL: {len(bad)} unexpected docs/p1_hardening references")
        for i, ln in bad:
            print(f"    line {i}: {ln[:160]}")
            issues.append(f"unexpected docs/p1_hardening at line {i}")
    print()

    # 8. Nine-row synthesis table (count rows in the §8.1 table)
    print("## Nine-row synthesis table at §8.1")
    m = re.search(r"### §8\.1[^\n]*\n.*?(?=\n### §8\.2)", text, flags=re.DOTALL)
    if m:
        block = m.group(0)
        # Count data rows (lines starting with "| 1 |", "| 2 |", ..., "| 9 |").
        rows = re.findall(r"^\|\s*\d+\s*\|", block, flags=re.MULTILINE)
        n_rows = len(rows)
        print(f"  Numbered rows: {n_rows} (expected: 9)")
        if n_rows != 9:
            issues.append(f"§8.1 has {n_rows} numbered rows, expected 9")
    else:
        print("  FAIL: §8.1 not found")
        issues.append("§8.1 not found")
    print()

    # Summary
    print("=" * 72)
    if issues:
        print(f"VERIFICATION ISSUES ({len(issues)}):")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
