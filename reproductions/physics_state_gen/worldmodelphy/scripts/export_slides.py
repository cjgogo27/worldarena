from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "slides_cn.md"
TARGET = ROOT / "docs" / "slides_cn.html"


def render_block(block: str) -> str:
    lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
    html: list[str] = []
    in_list = False
    for line in lines:
        if line.startswith("# "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("### "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h3>{escape(line[4:])}</h3>")
        elif line.startswith("- "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{escape(line[2:])}</li>")
        elif line.startswith(tuple(f"{i}." for i in range(10))):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p class='num'>{escape(line)}</p>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p>{escape(line)}</p>")
    if in_list:
        html.append("</ul>")
    return "\n".join(html)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    slides = [render_block(block) for block in text.split("\n---\n") if block.strip()]
    body = "\n".join(f"<section class='slide'>{slide}</section>" for slide in slides)
    html = f"""<!doctype html>
<html lang='zh'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>WorldModelPhy Slides</title>
  <style>
    :root {{ --mit-red: #a31f34; --ink: #111; --muted: #555; --bg: #f7f4ef; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', sans-serif; color: var(--ink); background: #ddd; }}
    .slide {{ width: 1280px; min-height: 720px; padding: 56px 72px; margin: 24px auto; background: var(--bg); position: relative; box-shadow: 0 10px 30px rgba(0,0,0,.18); overflow: hidden; }}
    .slide::before {{ content: ''; position: absolute; inset: 0 0 auto 0; height: 14px; background: var(--mit-red); }}
    h1 {{ font-size: 38px; margin: 0 0 20px; color: var(--mit-red); line-height: 1.2; }}
    h2 {{ font-size: 28px; margin: 22px 0 12px; color: #222; }}
    h3 {{ font-size: 22px; margin: 18px 0 8px; color: #222; }}
    p, li {{ font-size: 22px; line-height: 1.55; margin: 8px 0; }}
    ul {{ margin: 12px 0 0 18px; padding-left: 18px; }}
    .num {{ font-weight: 600; }}
    @media print {{ body {{ background: white; }} .slide {{ margin: 0; box-shadow: none; page-break-after: always; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    TARGET.write_text(html, encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
