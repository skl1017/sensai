"""Cut every demo into one reviewer-friendly video: demos/sensai-features.mp4.

    python3 demos/montage.py

For each demo, in catalogue order: a 2 s title card, then the recording with
its idle stretches (model thinking, pauses) shortened, and a banner under the
terminal naming the feature. The MP4 carries one chapter per feature, and
demos/sensai-features.md lists the same timecodes.

Needs ffmpeg/ffprobe on the PATH and Pillow (text is drawn with Pillow because
not every ffmpeg build has `drawtext`).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEMOS = Path(__file__).resolve().parent
OUT = DEMOS / "sensai-features.mp4"
INDEX = DEMOS / "sensai-features.md"

FEATURES = [
    ("b1-cli-chatbot", "B1", "Chatbot CLI : streaming, historique de session, erreurs propres, modèle choisi par config"),
    ("m1-sessions-profile", "M1", "Sessions persistantes et profil utilisateur injecté à chaque tour"),
    ("m2-token-budget", "M2", "Budget de tokens : les anciens tours sont résumés, les faits clés restent"),
    ("m3-structured-memory", "M3", "Mémoire structurée SQLite : l'agent enregistre, met à jour, oublie"),
    ("t1-mcp", "T1", "MCP : serveur filesystem existant + notre propre serveur (calculatrice)"),
    ("t2-code-sandbox", "T2", "Code exécuté en sandbox Docker, l'agent corrige sa propre erreur"),
    ("t3-web-search", "T3", "Recherche web (SearXNG) avec sources citées"),
    ("t4-file-permissions", "T4", "Accès fichiers sous permissions : data/ autorisé, le reste refusé"),
    ("r1-basic-rag", "R1", "RAG : documents locaux indexés, réponses avec sources"),
    ("r2-advanced-rag", "R2", "RAG avancé : reranking, seuil de score, filtres déduits de la question"),
    ("a1-react-loop", "A1", "Boucle ReAct : plusieurs appels d'outils enchaînés avant de répondre"),
    ("a2-multi-agent", "A2", "Multi-agents : délégation à un sous-agent tuteur (ask_tutor)"),
    ("a3-human-in-the-loop", "A3", "Human-in-the-loop : refus puis accord avant d'écrire un fichier"),
    ("a5-persona", "A5", "Persona changée en cours de session, historique conservé"),
    ("a6-semantic-cache", "A6", "Cache sémantique : même question, réponse servie sans le modèle"),
    ("a7-prompt-optimization", "A7", "Few-shot choisis selon la question + réécriture des questions de suivi"),
    ("ev2-guardrails", "EV2", "Guardrails : injections bloquées en entrée, PII masquées en sortie"),
    ("x2-branching-interrupt", "X2", "Branches (édition d'un ancien message) et interruption par Ctrl+C"),
    ("x4-ask-user", "X4", "Le modèle pose ses questions à l'utilisateur (choix multiple, texte libre)"),
]

W, H, BAR = 1400, 800, 80
FPS = 25
CARD_S, INTRO_S = 2.0, 4.0
HOLD_S, END_HOLD_S = 1.5, 3.0  # idle kept inside a clip / at its end
BG, BAR_BG, FG, MUTED, ACCENT = "#1e1e2e", "#11111b", "#cdd6f4", "#9399b2", "#89b4fa"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ENCODE = ["-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p", "-r", str(FPS)]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stderr


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(out)


def banner(path: Path, code: str, label: str, n: int) -> None:
    img = Image.new("RGB", (W, BAR), BAR_BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((24, 18, 24 + 86, BAR - 18), radius=8, fill=ACCENT)
    d.text((24 + 43, BAR // 2), code, font=font(24, True), fill=BAR_BG, anchor="mm")
    d.text((130, BAR // 2), label, font=font(22), fill=FG, anchor="lm")
    d.text((W - 24, BAR // 2), f"{n}/{len(FEATURES)}", font=font(20), fill=MUTED, anchor="rm")
    img.save(path)


def card(path: Path, title: str, subtitle: str, footer: str = "") -> None:
    img = Image.new("RGB", (W, H + BAR), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, (H + BAR) // 2 - 40), title, font=font(72, True), fill=ACCENT, anchor="mm")
    d.text((W // 2, (H + BAR) // 2 + 40), subtitle, font=font(28), fill=FG, anchor="mm")
    if footer:
        d.text((W // 2, H + BAR - 60), footer, font=font(20), fill=MUTED, anchor="mm")
    img.save(path)


def kept_ranges(clip: Path) -> list[tuple[float, float]]:
    """Clip time ranges to keep: everything but idle beyond HOLD_S (END_HOLD_S at the end)."""
    total = duration(clip)
    log = run("ffmpeg", "-hide_banner", "-i", str(clip), "-vf", "freezedetect=n=0.001:d=1",
              "-map", "0:v", "-f", "null", "-")
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", log)]
    ends += [total] * (len(starts) - len(ends))  # frozen until the end

    keep, cursor = [], 0.0
    for start, end in zip(starts, ends, strict=True):
        hold = END_HOLD_S if end >= total - 0.05 else HOLD_S
        if start > cursor:
            keep.append((cursor, start))
        keep.append((start, min(end, start + hold)))
        cursor = end
    if cursor < total:
        keep.append((cursor, total))
    return keep


def render_clip(clip: Path, banner_png: Path, out: Path) -> None:
    select = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in kept_ranges(clip))
    graph = (
        f"[0:v]select='{select}',setpts=N/{FPS}/TB,pad={W}:{H + BAR}:0:0:color={BAR_BG}[v];"
        f"[v][1:v]overlay=0:{H}"
    )
    run("ffmpeg", "-y", "-i", str(clip), "-i", str(banner_png), "-filter_complex", graph,
        "-an", *ENCODE, str(out))


def render_still(png: Path, seconds: float, out: Path) -> None:
    run("ffmpeg", "-y", "-loop", "1", "-t", str(seconds), "-i", str(png), "-an", *ENCODE, str(out))


def stamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def main() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on the PATH")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        parts: list[Path] = []
        chapters: list[tuple[str, str, float, float]] = []  # code, label, start, end
        t = 0.0

        card(tmp / "intro.png", "Sensai", f"{len(FEATURES)} features en démonstration",
             "Une séquence par feature, titrée en bas de l'écran. Chapitres intégrés à la vidéo.")
        render_still(tmp / "intro.png", INTRO_S, tmp / "000-intro.mp4")
        parts.append(tmp / "000-intro.mp4")
        t += duration(parts[-1])

        for n, (name, code, label) in enumerate(FEATURES, start=1):
            print(f"[{n}/{len(FEATURES)}] {name}", flush=True)
            start = t
            card(tmp / f"{n:03d}-card.png", code, label)
            render_still(tmp / f"{n:03d}-card.png", CARD_S, tmp / f"{n:03d}-a-card.mp4")
            banner(tmp / f"{n:03d}-banner.png", code, label, n)
            render_clip(DEMOS / f"{name}.mp4", tmp / f"{n:03d}-banner.png", tmp / f"{n:03d}-b-clip.mp4")
            for part in (tmp / f"{n:03d}-a-card.mp4", tmp / f"{n:03d}-b-clip.mp4"):
                parts.append(part)
                t += duration(part)
            chapters.append((code, label, start, t))

        (tmp / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        meta = [";FFMETADATA1", "title=Sensai — démonstration des features"]
        for code, label, start, end in chapters:
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}",
                     f"END={int(end * 1000)}", f"title={code} — {label}"]
        (tmp / "meta.txt").write_text("\n".join(meta) + "\n")
        run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
            "-i", str(tmp / "meta.txt"), "-map_metadata", "1", "-map_chapters", "1",
            "-c", "copy", "-movflags", "+faststart", str(OUT))

    lines = [
        "# Sensai — vidéo de démonstration",
        "",
        f"`sensai-features.mp4` ({stamp(t)}) enchaîne les {len(FEATURES)} démos, attentes "
        "raccourcies. Chaque séquence a un chapitre dans la vidéo ; les mêmes timecodes :",
        "",
        "| Début | Feature | Ce qu'on voit |",
        "|---|---|---|",
    ]
    lines += [f"| {stamp(start)} | **{code}** | {label} |" for code, label, start, _ in chapters]
    INDEX.write_text("\n".join(lines) + "\n")
    print(f"{OUT} ({stamp(t)})\n{INDEX}")


if __name__ == "__main__":
    main()
