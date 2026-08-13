#!/usr/bin/env python3
"""Show that TEDS treats rowspan/colspan attribute order as equivalent."""

from __future__ import annotations

import sys
from pathlib import Path

from lxml import html


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(ROOT / "repo" / "TFLOP"))


HTML_A = """
<html><body><table>
  <tr>
    <td colspan="2" rowspan="2">A</td>
    <td>B</td>
  </tr>
  <tr>
    <td>C</td>
  </tr>
</table></body></html>
"""

HTML_B = """
<html><body><table>
  <tr>
    <td rowspan="2" colspan="2">A</td>
    <td>B</td>
  </tr>
  <tr>
    <td>C</td>
  </tr>
</table></body></html>
"""

HTML_C = """
<html><body><table>
  <tr>
    <td colspan="2">A</td>
    <td>B</td>
  </tr>
  <tr>
    <td>C</td>
  </tr>
</table></body></html>
"""


def show_first_cell_attributes(name: str, html_text: str) -> None:
    root = html.fromstring(html_text)
    first_td = root.xpath("body/table//td")[0]
    print(f"{name}: lxml attributes = {dict(first_td.attrib)}")
    print(
        f"{name}: TEDS reads colspan={int(first_td.attrib.get('colspan', '1'))}, "
        f"rowspan={int(first_td.attrib.get('rowspan', '1'))}"
    )


def main() -> None:
    show_first_cell_attributes("HTML_A", HTML_A)
    show_first_cell_attributes("HTML_B", HTML_B)
    show_first_cell_attributes("HTML_C", HTML_C)

    try:
        from tflop.evaluator import TEDS
    except ModuleNotFoundError as exc:
        print()
        print(f"Cannot run full TEDS in this Python env: missing {exc.name!r}")
        print("The attribute-order conclusion above is still the same TEDS logic.")
        return

    metric = TEDS(structure_only=False)
    metric_s = TEDS(structure_only=True)
    print()
    print("A vs B, only attribute order differs")
    print("TEDS  =", metric.evaluate(HTML_A, HTML_B))
    print("TEDS-S=", metric_s.evaluate(HTML_A, HTML_B))
    print()
    print("A vs C, rowspan differs")
    print("TEDS  =", metric.evaluate(HTML_A, HTML_C))
    print("TEDS-S=", metric_s.evaluate(HTML_A, HTML_C))


if __name__ == "__main__":
    main()
