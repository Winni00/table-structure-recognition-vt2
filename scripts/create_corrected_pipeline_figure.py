#!/usr/bin/env python3
"""Create the report diagram for the target-domain TFLOP workflow."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2800
HEIGHT = 1050
SCALE = 1

OUT_DIR = Path(__file__).resolve().parents[1] / "report_feedback_revision_current" / "figures"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str = "",
) -> None:
    x0, y0, x1, y1 = box
    title_font = font(26, bold=True)
    sub_font = font(22)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_box[2] - title_box[0]
    title_h = title_box[3] - title_box[1]
    if subtitle:
        sub_box = draw.multiline_textbbox(
            (0, 0), subtitle, font=sub_font, spacing=6, align="center"
        )
        sub_w = sub_box[2] - sub_box[0]
        sub_h = sub_box[3] - sub_box[1]
        total_h = title_h + 14 + sub_h
        title_y = y0 + (y1 - y0 - total_h) / 2
        draw.text(((x0 + x1 - title_w) / 2, title_y), title, font=title_font, fill="#111111")
        draw.multiline_text(
            ((x0 + x1 - sub_w) / 2, title_y + title_h + 14),
            subtitle,
            font=sub_font,
            fill="#222222",
            spacing=6,
            align="center",
        )
    else:
        draw.text(
            ((x0 + x1 - title_w) / 2, y0 + (y1 - y0 - title_h) / 2),
            title,
            font=title_font,
            fill="#111111",
        )


def node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    title: str,
    subtitle: str,
) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline="#333333", width=3)
    centered_text(draw, box, title, subtitle)


def arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    label: str = "",
    label_xy: tuple[int, int] | None = None,
) -> None:
    draw.line(points, fill="#333333", width=5, joint="curve")
    x2, y2 = points[-1]
    x1, y1 = points[-2]
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        head = [(x2, y2), (x2 - 18 * direction, y2 - 11), (x2 - 18 * direction, y2 + 11)]
    else:
        direction = 1 if y2 > y1 else -1
        head = [(x2, y2), (x2 - 11, y2 - 18 * direction), (x2 + 11, y2 - 18 * direction)]
    draw.polygon(head, fill="#333333")
    if label and label_xy:
        label_font = font(23)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        pad = 7
        lx, ly = label_xy
        draw.rounded_rectangle(
            (lx - pad, ly - pad, lx + bbox[2] + pad, ly + bbox[3] + pad),
            radius=5,
            fill="#ffffff",
        )
        draw.text((lx, ly), label, font=label_font, fill="#222222")


def dashed_rectangle(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], dash: int = 18, gap: int = 11
) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash + gap):
        draw.line((x, y0, min(x + dash, x1), y0), fill="#666666", width=3)
        draw.line((x, y1, min(x + dash, x1), y1), fill="#666666", width=3)
    for y in range(y0, y1, dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1)), fill="#666666", width=3)
        draw.line((x1, y, x1, min(y + dash, y1)), fill="#666666", width=3)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    groups = [
        ((40, 95, 885, 540), "Input preparation"),
        ((925, 95, 1765, 540), "Text extraction"),
        ((1805, 95, 2760, 540), "Table reconstruction"),
    ]
    for box, heading in groups:
        draw.rounded_rectangle(box, radius=16, outline="#555555", width=3)
        heading_box = draw.textbbox((0, 0), heading, font=font(31, bold=True))
        draw.text(
            ((box[0] + box[2] - heading_box[2]) / 2, 35),
            heading,
            font=font(31, bold=True),
            fill="#111111",
        )

    crop = (90, 205, 420, 405)
    orientation = (505, 205, 835, 405)
    psenet = (975, 205, 1305, 405)
    master = (1385, 205, 1715, 405)
    tflop = (1855, 180, 2245, 430)
    html = (2325, 205, 2710, 405)

    node(draw, crop, "#f2f4f7", "Provided table crop", "Input image")
    node(draw, orientation, "#dceaf7", "Orientation correction", "Rotate if required")
    node(draw, psenet, "#dff1e8", "PSENet", "Detect text-region boxes")
    node(draw, master, "#fff0cc", "MASTER", "Recognise region text")
    node(
        draw,
        tflop,
        "#f7ded7",
        "TFLOP",
        "Inputs: oriented crop,\nPSENet boxes, MASTER text",
    )
    node(
        draw,
        html,
        "#e7def2",
        "OTSL + HTML",
        "Structure, cell assignments,\nand reconstructed table",
    )

    arrow(draw, [(420, 305), (505, 305)])
    arrow(draw, [(835, 305), (975, 305)], "oriented crop", (842, 255))
    arrow(draw, [(1305, 305), (1385, 305)], "regions", (1302, 255))
    arrow(
        draw,
        [(1715, 305), (1855, 305)],
        "crop + boxes + text",
        (1670, 255),
    )
    arrow(draw, [(2245, 305), (2325, 305)])

    eval_group = (1510, 640, 2760, 995)
    dashed_rectangle(draw, eval_group)
    eval_heading = "Optional offline evaluation (requires reference HTML)"
    eval_heading_box = draw.textbbox((0, 0), eval_heading, font=font(29, bold=True))
    draw.text(
        ((eval_group[0] + eval_group[2] - eval_heading_box[2]) / 2, 580),
        eval_heading,
        font=font(29, bold=True),
        fill="#333333",
    )

    prediction = (1570, 750, 1925, 895)
    reference = (1990, 750, 2345, 895)
    teds = (2410, 750, 2700, 895)
    node(draw, prediction, "#e7def2", "Predicted HTML", "Pipeline output")
    node(draw, reference, "#f2f4f7", "Reference HTML", "Evaluation only")
    node(draw, teds, "#e8edf2", "TEDS / TEDS-S", "Reported scores")
    arrow(draw, [(1925, 822), (1960, 822), (1960, 725), (2368, 725), (2368, 795), (2410, 795)])
    arrow(draw, [(2345, 850), (2410, 850)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_DIR / "pipeline_overview_corrected.png", dpi=(300, 300))
    image.save(OUT_DIR / "pipeline_overview_corrected.pdf", "PDF", resolution=300.0)


if __name__ == "__main__":
    main()
