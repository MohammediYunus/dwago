"""dwago map — the codebase as a spiral galaxy, in one self-contained page.

Direction (user-set): photographic realism, Hubble-image reference. That decides
almost everything:

*3D only.* The earlier hybrid kept a canvas-2D view whose pan/zoom worked by
blitting cached layers and re-rendering on a debounce — the visible seam ("zoom
glitches, feels delayed") was structural, not tunable. Here every frame is drawn
fresh by the GPU and zoom is a uniform: there is nothing to reload, so there is
nothing to glitch.

*The graph maps onto galactic anatomy instead of abstract blobs.* The largest
connected community is the warm core bulge. The rest are star clusters strung
along two logarithmic spiral arms, elongated along the arm tangent the way real
star-forming regions are. Disconnected material (isolated doc clusters — a
third of a real corpus) becomes the sparse stellar halo. Dust lanes trace the
arms' inner edges; a generated backdrop supplies field stars, twinkle, faint
distant galaxies, and diffraction spikes on the brightest hubs.

*Edges are annotation, not scenery.* A photograph of a galaxy has no lines in
it. Structural and co-change links appear only as an overlay — thin
mission-control style constellation lines — when a star is selected or blast is
armed. Realism at rest, data on demand.

*No webfont.* The chrome uses the system stack: zero bytes, zero FOUT, and the
neutral instrument-panel look actual mission UIs have.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["write_html", "collect_graph_data"]

GOLDEN = 2.399963229728653


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _merge_communities(comms: np.ndarray, src: np.ndarray, dst: np.ndarray,
                       top_k: int = 36, passes: int = 4):
    """Collapse micro-communities into their most-connected large neighbour.

    Leiden on a real repository produces ~1,000 communities with power-law
    sizes; rendered raw they are confetti. Assignment iterates so chains of
    micro-communities resolve through each other to a kept group; whatever
    still has no connection to anything assigned lands in one 'misc' bucket,
    which at that point is genuinely isolated material. Presentational only —
    the raw community id still ships for the detail panel.
    """
    from collections import defaultdict

    ids, counts = np.unique(comms, return_counts=True)
    keep = set(ids[np.argsort(-counts)][:top_k].tolist())

    pair_mass: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for a, b in zip(src, dst):
        ca, cb = int(comms[a]), int(comms[b])
        if ca == cb:
            continue
        pair_mass[ca][cb] += 1.0
        pair_mass[cb][ca] += 1.0

    group: dict[int, int | None] = {int(c): (int(c) if int(c) in keep else None)
                                    for c in ids.tolist()}
    for _ in range(passes):
        changed = False
        for cid, g in group.items():
            if g is not None:
                continue
            votes: dict[int, float] = defaultdict(float)
            for nb, w in pair_mass.get(cid, {}).items():
                tgt = group.get(nb)
                if tgt is not None:
                    votes[tgt] += w
            if votes:
                group[cid] = max(votes.items(), key=lambda kv: kv[1])[0]
                changed = True
        if not changed:
            break

    misc = max(int(ids.max()), 0) + 1
    out = np.array([group[int(c)] if group[int(c)] is not None else misc
                    for c in comms], dtype=np.int32)
    remap = {cid: i for i, cid in enumerate(sorted(set(out.tolist())))}
    merged = np.array([remap[int(c)] for c in out], dtype=np.int32)
    return merged, remap.get(misc, -1)


# ── galaxy morphology ────────────────────────────────────────────────────────
#
# Two logarithmic arms, r = R0 · e^(B·θ), matching the grand-design spiral of
# the reference image. Communities are placed along the arms biggest-first;
# each cluster is elongated along the local arm tangent, the way real
# star-forming regions smear along their arm.

ARM_R0 = 90.0
ARM_B = 0.185
ARM_STEP = 0.42          # radians of arm per community slot
ARM_START = 0.15
DISC_R = 380.0           # nominal outer radius, drives color temperature


def _arm_point(slot: int, arm: int) -> tuple[float, float, float]:
    """Center of community slot `slot` on arm `arm` (0/1): (x, z, theta)."""
    th = ARM_START + slot * ARM_STEP
    r = ARM_R0 * math.exp(ARM_B * th)
    a = th + arm * math.pi
    return r * math.cos(a), r * math.sin(a), a


def _galaxy_layout(gcomm: np.ndarray, degrees: np.ndarray,
                   rng: np.random.RandomState, misc_id: int):
    """Positions (n, 3): XZ is the galactic plane, Y is disc thickness."""
    ids, counts = np.unique(gcomm, return_counts=True)
    mask = ids != misc_id
    order = ids[mask][np.argsort(-counts[mask])]

    # cluster parameters per community
    kind = {}                 # cid -> ('core', ...) | ('arm', cx, cz, tangent, sigma)
    for k, cid in enumerate(order.tolist()):
        n = int(counts[ids == cid][0])
        if k == 0:
            kind[cid] = ("core", 52.0)
        else:
            slot, arm = (k - 1) // 2, (k - 1) % 2
            cx, cz, a = _arm_point(slot, arm)
            tangent = a + math.pi / 2 + ARM_B      # spiral tangent direction
            sigma = min(11.0 + 2.6 * math.sqrt(n), 40.0)
            kind[cid] = ("arm", cx, cz, tangent, sigma)

    n = len(gcomm)
    pos = np.zeros((n, 3), dtype=np.float32)
    dmax = max(float(degrees.max()), 1.0)
    for i in range(n):
        cid = int(gcomm[i])
        pull = 1.0 - 0.55 * math.sqrt(degrees[i] / dmax)
        if cid == misc_id:
            # stellar halo: sparse sphere around the disc, vertically squashed
            v = rng.normal(0, 1, 3)
            v /= max(np.linalg.norm(v), 1e-6)
            r = rng.uniform(260.0, 520.0)
            pos[i] = (v[0] * r, v[1] * r * 0.55, v[2] * r)
            continue
        spec = kind[cid]
        if spec[0] == "core":
            s = spec[1] * pull
            pos[i] = (rng.normal(0, s), rng.normal(0, s * 0.45), rng.normal(0, s))
        else:
            _, cx, cz, tang, sigma = spec
            u = rng.normal(0, sigma * 1.75) * pull      # along the arm
            v = rng.normal(0, sigma * 0.72) * pull      # across it
            pos[i] = (cx + u * math.cos(tang) - v * math.sin(tang),
                      rng.normal(0, 9.0),
                      cz + u * math.sin(tang) + v * math.cos(tang))
    return pos


def _star_colors(pos: np.ndarray, gcomm: np.ndarray, misc_id: int,
                 rng: np.random.RandomState) -> np.ndarray:
    """Natural astro palette by galactocentric distance.

    Core: warm K-giant amber. Arms: blue-white OB associations with a few pink
    HII regions. Halo: dim neutral. Matches how a real spiral photographs.
    """
    warm = np.array([255, 208, 158], dtype=np.float32)
    cream = np.array([255, 240, 222], dtype=np.float32)
    blue = np.array([135, 172, 255], dtype=np.float32)
    pink = np.array([255, 148, 186], dtype=np.float32)
    halo = np.array([196, 202, 214], dtype=np.float32)

    n = len(gcomm)
    out = np.zeros((n, 3), dtype=np.uint8)
    d = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
    t = np.clip(d / DISC_R, 0, 1)
    jit = rng.normal(0, 9, (n, 3))
    hii = rng.rand(n)
    for i in range(n):
        if gcomm[i] == misc_id:
            c = halo
        elif t[i] < 0.32:
            c = warm + (cream - warm) * (t[i] / 0.32)
        else:
            u = (t[i] - 0.32) / 0.68
            c = cream + (blue - cream) * u
            if hii[i] < 0.05:
                c = pink
        out[i] = np.clip(c + jit[i], 0, 255).astype(np.uint8)
    return out


def _arm_glow(rng: np.random.RandomState):
    """Soft luminous haze tracing the arm bodies. (m, 4): x, y, z, size.

    This is what actually makes the spiral legible: the arms of a real galaxy
    read as continuous glow, and the dust lanes read as dark bites out of that
    glow — without it the dust just erases background starfield and looks like
    a void. Drawn additively beneath the dust pass.
    """
    rows = []
    for arm in (0, 1):
        th = ARM_START - 0.1
        while th < ARM_START + 17.2 * ARM_STEP:
            r = ARM_R0 * math.exp(ARM_B * th)
            a = th + arm * math.pi
            rows.append((r * math.cos(a) + rng.normal(0, 7),
                         rng.normal(0, 3.0),
                         r * math.sin(a) + rng.normal(0, 7),
                         38 + r * 0.042))
            th += 0.034
    return np.array(rows, dtype=np.float32)


def _dust_lanes(rng: np.random.RandomState):
    """Dark dust sprites hugging the arms' inner edges. (m, 4): x, y, z, size."""
    rows = []
    for arm in (0, 1):
        th = ARM_START - 0.15
        while th < ARM_START + 18 * ARM_STEP:
            r = ARM_R0 * math.exp(ARM_B * th) * 0.93     # inner edge
            a = th + arm * math.pi
            for _ in range(2):
                rows.append((
                    r * math.cos(a) + rng.normal(0, 9),
                    rng.normal(0, 4.0),
                    r * math.sin(a) + rng.normal(0, 9),
                    rng.uniform(22, 48),
                ))
            th += 0.055
    return np.array(rows, dtype=np.float32)


def _arm_filler(rng: np.random.RandomState):
    """Thousands of tiny unresolved stars tracing the arms and bulge.

    The data nodes alone are ~14k points — a real spiral photograph reads as
    continuous luminous texture because millions of unresolved stars fill the
    arms. This filler is decoration only (never pickable): it gives the arms
    their visible spiral shape and gives the dust lanes something to occlude.
    Returns (pos (m,3), color (m,3) uint8, size (m,)).
    """
    pts, cols, sizes = [], [], []
    warm = np.array([255, 208, 158]); blue = np.array([140, 178, 255])
    cream = np.array([250, 238, 224])

    # arms: sampled proportional to ARC LENGTH, not uniform in theta — a log
    # spiral covers ever more distance per radian as r grows, so uniform-theta
    # sampling starves the outer arms and leaves a dark annulus mid-disc
    # (observed directly). Rejection-sample against r/r_max.
    th_lo, th_hi = ARM_START - 0.2, ARM_START + 17.5 * ARM_STEP
    r_max = ARM_R0 * math.exp(ARM_B * th_hi)
    for arm in (0, 1):
        got = 0
        while got < 6800:
            th = rng.uniform(th_lo, th_hi)
            if rng.rand() > (ARM_R0 * math.exp(ARM_B * th)) / r_max:
                continue
            got += 1
            r = ARM_R0 * math.exp(ARM_B * th)
            a = th + arm * math.pi
            tang = a + math.pi / 2 + ARM_B
            u = rng.normal(0, 34.0)
            v = rng.normal(0, 12.0 + r * 0.055)
            x = r * math.cos(a) + u * math.cos(tang) - v * math.sin(tang)
            z = r * math.sin(a) + u * math.sin(tang) + v * math.cos(tang)
            pts.append((x, rng.normal(0, 6.5), z))
            t = min(r / DISC_R, 1.0)
            c = cream + (blue - cream) * t + rng.normal(0, 10, 3)
            cols.append(np.clip(c, 0, 255))
            sizes.append(rng.uniform(1.3, 3.4))

    # bulge: dense warm core
    for _ in range(5200):
        rr = abs(rng.normal(0, 60.0))
        an = rng.uniform(0, 6.283)
        pts.append((rr * math.cos(an), rng.normal(0, 16.0), rr * math.sin(an)))
        c = warm + rng.normal(0, 12, 3)
        cols.append(np.clip(c, 0, 255))
        sizes.append(rng.uniform(1.0, 2.8))

    return (np.array(pts, dtype=np.float32),
            np.array(cols, dtype=np.uint8),
            np.array(sizes, dtype=np.float32))


def _nebula(rng: np.random.RandomState):
    """Faint dust clouds behind everything — the wispy nebulosity real deep-sky
    photos have instead of pure black. A few clustered complexes in muted rust,
    blue and teal, plus a broad band suggesting foreground galactic dust.
    Returns (pos (m,3), color (m,3) uint8, size (m,), alpha (m,))."""
    pts, cols, sizes, alphas = [], [], [], []
    palettes = [
        np.array([120, 88, 72]),    # rust
        np.array([62, 82, 138]),    # deep blue
        np.array([56, 102, 112]),   # teal
        np.array([96, 84, 92]),     # dusty mauve
    ]
    # cloud complexes
    for c in range(12):
        v = rng.normal(0, 1, 3); v /= max(np.linalg.norm(v), 1e-6)
        center = v * rng.uniform(1700, 2300)
        base = palettes[c % len(palettes)]
        for _ in range(26):
            off = rng.normal(0, 260, 3)
            pts.append(center + off)
            cols.append(np.clip(base + rng.normal(0, 12, 3), 0, 255))
            sizes.append(rng.uniform(160, 430))
            alphas.append(rng.uniform(0.06, 0.11))
    # broad diagonal dust band
    axis = np.array([0.8, 0.35, 0.5]); axis /= np.linalg.norm(axis)
    for _ in range(70):
        t = rng.uniform(-1, 1)
        perp = rng.normal(0, 0.16, 3)
        d = axis * t + perp
        d /= max(np.linalg.norm(d), 1e-6)
        pts.append(d * rng.uniform(1900, 2400))
        base = palettes[0] if rng.rand() < 0.6 else palettes[3]
        cols.append(np.clip(base + rng.normal(0, 10, 3), 0, 255))
        sizes.append(rng.uniform(170, 420))
        alphas.append(rng.uniform(0.04, 0.075))
    return (np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8),
            np.array(sizes, dtype=np.float32), np.array(alphas, dtype=np.float32))


def _backdrop(rng: np.random.RandomState):
    """Field stars and distant galaxies on a far shell."""
    nb = 3800
    v = rng.normal(0, 1, (nb, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    bg = v * rng.uniform(2300, 2900, (nb, 1)).astype(np.float32)
    bg_size = rng.uniform(0.7, 2.4, nb).astype(np.float32)
    # a sprinkling of brighter foreground-ish stars
    bright = rng.rand(nb) < 0.02
    bg_size[bright] *= rng.uniform(2.2, 3.6, int(bright.sum()))
    bg_phase = rng.uniform(0, 6.283, nb).astype(np.float32)
    # temperature scatter: most white, some warm, some blue
    tt = rng.rand(nb)
    bg_col = np.full((nb, 3), 235, dtype=np.uint8)
    bg_col[tt < 0.22] = (255, 214, 176)
    bg_col[tt > 0.80] = (198, 214, 255)

    ng = 90
    gv = rng.normal(0, 1, (ng, 3)).astype(np.float32)
    gv /= np.linalg.norm(gv, axis=1, keepdims=True)
    gal = gv * rng.uniform(2200, 2700, (ng, 1)).astype(np.float32)
    gal_size = rng.uniform(6, 17, ng).astype(np.float32)
    gal_size[rng.rand(ng) < 0.08] *= 2.2
    gal_angle = rng.uniform(0, 3.1416, ng).astype(np.float32)
    gal_ell = rng.uniform(0.30, 0.85, ng).astype(np.float32)
    gt = rng.rand(ng)
    gal_col = np.full((ng, 3), 210, dtype=np.uint8)
    gal_col[gt < 0.4] = (226, 206, 180)
    gal_col[gt > 0.7] = (185, 198, 232)
    return (bg, bg_size, bg_phase, bg_col), (gal, gal_size, gal_angle, gal_ell, gal_col)


# ── data ─────────────────────────────────────────────────────────────────────

def collect_graph_data(store, max_nodes: int = 250_000) -> dict:
    nodes = list(store.conn.execute(
        "SELECT idx, label, kind, source_file, start_line, community "
        "FROM nodes ORDER BY idx"))
    if max_nodes and len(nodes) > max_nodes:
        log.warning("graph has %d nodes; rendering the first %d", len(nodes), max_nodes)
        nodes = nodes[:max_nodes]
    keep = {r["idx"] for r in nodes}
    remap = {r["idx"]: i for i, r in enumerate(nodes)}
    n = len(nodes)

    files = {r["path"]: dict(r) for r in store.conn.execute(
        "SELECT path, hotspot, bus_factor, primary_owner, last_modified FROM files")}

    labels, kinds, paths = [], [], []
    comms = np.zeros(n, dtype=np.int32)
    hotspot = np.zeros(n, dtype=np.float32)
    age = np.zeros(n, dtype=np.int32)
    bus = np.zeros(n, dtype=np.int32)
    owner_idx = np.zeros(n, dtype=np.int32)
    lines = np.zeros(n, dtype=np.int32)
    owners: dict[str, int] = {}
    for i, r in enumerate(nodes):
        labels.append((r["label"] or "")[:120])
        kinds.append(r["kind"] or "")
        pth = r["source_file"] or ""
        paths.append(pth)
        comms[i] = int(r["community"]) if r["community"] is not None else -1
        lines[i] = int(r["start_line"] or 0)
        f = files.get(pth)
        o = ""
        if f:
            hotspot[i] = float(f["hotspot"] or 0.0)
            age[i] = int(f["last_modified"] or 0)
            bus[i] = int(f["bus_factor"] or 0)
            o = f["primary_owner"] or ""
        if o not in owners:
            owners[o] = len(owners)
        owner_idx[i] = owners[o]

    src, dst, chan = [], [], []
    for e in store.conn.execute("SELECT src, dst, channel FROM edges"):
        a, b = e["src"], e["dst"]
        if a in keep and b in keep:
            src.append(remap[a]); dst.append(remap[b])
            chan.append(0 if e["channel"] == "structural" else 1)
    src = np.array(src, dtype=np.int32)
    dst = np.array(dst, dtype=np.int32)
    chan = np.array(chan, dtype=np.int32)

    deg = np.zeros(n, dtype=np.int32)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)

    gcomm, misc_id = _merge_communities(comms, src, dst)
    rng = np.random.RandomState(1337)
    pos = _galaxy_layout(gcomm, deg.astype(np.float32), rng, misc_id)
    natural = _star_colors(pos, gcomm, misc_id, rng)
    dust = _dust_lanes(rng)
    glow = _arm_glow(rng)
    fil_pos, fil_col, fil_size = _arm_filler(rng)
    (bg, bg_size, bg_phase, bg_col), (gal, gal_size, gal_angle, gal_ell, gal_col) = _backdrop(rng)
    neb_pos, neb_col, neb_size, neb_alpha = _nebula(rng)

    # comet routes: a few structural edges for the slow "satellite" motes
    struct = np.where(chan == 0)[0]
    rng2 = np.random.RandomState(7)
    routes = struct[rng2.choice(len(struct), size=min(400, len(struct)),
                                replace=False)].astype(np.int32) if len(struct) else struct

    return {
        "n": n, "miscId": misc_id,
        "labels": labels, "kinds": kinds, "paths": paths, "owners": list(owners),
        "community": _b64(comms), "gcomm": _b64(gcomm),
        "hotspot": _b64(hotspot), "age": _b64(age), "bus": _b64(bus),
        "ownerIdx": _b64(owner_idx), "line": _b64(lines), "deg": _b64(deg),
        "src": _b64(src), "dst": _b64(dst), "chan": _b64(chan),
        "pos": _b64(pos), "natural": _b64(natural), "routes": _b64(routes),
        "dust": _b64(dust), "glow": _b64(glow),
        "fil": _b64(fil_pos), "filCol": _b64(fil_col), "filSize": _b64(fil_size),
        "bg": _b64(bg), "bgSize": _b64(bg_size), "bgPhase": _b64(bg_phase), "bgCol": _b64(bg_col),
        "gal": _b64(gal), "galSize": _b64(gal_size), "galAngle": _b64(gal_angle),
        "galEll": _b64(gal_ell), "galCol": _b64(gal_col),
        "neb": _b64(neb_pos), "nebCol": _b64(neb_col),
        "nebSize": _b64(neb_size), "nebAlpha": _b64(neb_alpha),
    }


def write_html(store, out: Path, *, title: str = "dwago",
               max_nodes: int = 250_000) -> int:
    import html as _h

    data = collect_graph_data(store, max_nodes=max_nodes)
    # Escape script delimiters and comment markers without changing decoded text.
    replacements = {
        "__TITLE_JS__": json.dumps(title).replace("<", "\\u003c"),
        "__TITLE__": _h.escape(title),
        "__DATA__": json.dumps(data).replace("<", "\\u003c"),
    }
    # Replace once so marker-like text in a title or label stays literal.
    html = re.sub(r"__TITLE_JS__|__TITLE__|__DATA__",
                  lambda match: replacements[match.group()], _TEMPLATE)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return data["n"]


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · dwago</title>
<style>
:root {
  --space:#03050b; --ink:#dbe3f2; --dim:#8b95ab; --faint:#5a6478;
  --glass:rgba(10,15,28,.58); --brd:rgba(150,180,255,.13);
  --accent:#8ec6ff; --amber:#ffbe72; --alert:#ff7a6e;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}
* { box-sizing:border-box; margin:0; padding:0; }
html,body { height:100%; overflow:hidden; }
body { background:var(--space); color:var(--ink); font:400 13px/1.5 var(--sans); }
canvas { display:block; }
#gl, #hud { position:fixed; inset:0; width:100%; height:100%; touch-action:none; }
#hud { pointer-events:none; }

.panel {
  position:fixed; z-index:10; background:var(--glass); border:1px solid var(--brd);
  border-radius:12px; backdrop-filter:blur(18px) saturate(1.3);
  -webkit-backdrop-filter:blur(18px) saturate(1.3);
  box-shadow:inset 0 1px 0 rgba(190,215,255,.10), 0 12px 40px rgba(0,0,0,.5);
}
.plabel {
  font-size:10px; font-weight:600; letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); margin-bottom:8px;
}

#brand { top:16px; left:16px; padding:12px 16px; }
#brand .word { font:600 17px/1 var(--sans); letter-spacing:.16em; text-transform:uppercase; }
#brand .word b { color:var(--accent); font-weight:600; }
#brand .sub { font:400 10.5px var(--mono); color:var(--dim); margin-top:5px; }

#searchp { top:16px; left:50%; transform:translateX(-50%); width:min(440px,60vw); padding:10px 13px; }
#q { width:100%; border:none; outline:none; background:transparent;
     font:400 13.5px var(--sans); color:var(--ink); }
#q::placeholder { color:var(--faint); }
#results { max-height:260px; overflow-y:auto; margin-top:7px; display:none; }
#results.open { display:block; border-top:1px solid var(--brd); padding-top:7px; }
.hit { padding:5px 8px; border-radius:7px; cursor:pointer; white-space:nowrap;
       overflow:hidden; text-overflow:ellipsis; font-size:12.5px; }
.hit:hover, .hit.sel { background:rgba(142,198,255,.14); }
.hit small { color:var(--faint); margin-left:8px; font-size:10.5px; font-family:var(--mono); }

#ctrl { bottom:16px; left:16px; padding:14px 16px; width:242px; }
#ctrl .row { display:flex; gap:7px; margin-bottom:11px; }
#ctrl .row:last-child { margin-bottom:0; }
button, select {
  font:500 11.5px var(--sans); color:var(--ink); background:rgba(150,180,255,.07);
  border:1px solid var(--brd); border-radius:8px; padding:6px 10px; cursor:pointer;
}
button:hover { border-color:var(--accent); }
button.on { background:var(--accent); border-color:var(--accent); color:#04121f; }
select { flex:1; outline:none; appearance:none; }
select option { background:#0b1020; }
.grow { flex:1; }
input[type=range] { width:100%; accent-color:var(--accent); height:20px; }
#timelabel { font:400 11px var(--sans); color:var(--dim); margin-top:3px; min-height:15px; }

#detail { top:16px; right:16px; width:min(330px,86vw); max-height:calc(100vh - 32px);
          padding:16px 18px; overflow-y:auto; display:none; }
#detail.open { display:block; }
#detail h3 { font:600 14.5px var(--sans); word-break:break-all; margin-bottom:3px; }
#detail .loc { font:400 10.5px var(--mono); color:var(--dim); word-break:break-all; margin-bottom:10px; }
.pill { display:inline-block; font:500 10px var(--sans); letter-spacing:.06em;
        text-transform:uppercase; border:1px solid var(--brd); border-radius:99px;
        padding:2px 9px; margin:0 4px 4px 0; color:var(--dim); }
.pill.hot { color:var(--amber); border-color:rgba(255,190,114,.5); }
.pill.risk { color:var(--alert); border-color:rgba(255,122,110,.5); }
.nb { padding:4px 7px; border-radius:6px; cursor:pointer; font-size:12.5px;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.nb:hover { background:rgba(142,198,255,.12); }
.nb .tag { color:var(--faint); font-family:var(--mono); font-size:10px; margin-right:6px; }
.nb .tag.tmp { color:var(--amber); }
#detail .close { position:absolute; top:9px; right:11px; border:none; background:none;
                 font-size:15px; color:var(--faint); cursor:pointer; padding:4px; }
#detail .close:hover { color:var(--ink); }

#status { bottom:16px; right:16px; padding:9px 14px; font:400 10.5px var(--mono);
          color:var(--dim); min-width:210px; text-align:right; }
#status b { color:var(--ink); font-weight:500; }

#tip { position:fixed; pointer-events:none; display:none; z-index:20;
       background:rgba(8,12,22,.92); border:1px solid var(--brd); color:var(--ink);
       font:400 12px var(--sans); padding:5px 10px; border-radius:7px; max-width:340px;
       white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#tip small { color:var(--dim); margin-left:7px; font-family:var(--mono); font-size:10px; }

@media (max-width:720px) {
  #searchp { top:72px; left:16px; right:16px; transform:none; width:auto; }
  #detail { top:auto; bottom:86px; max-height:42vh; }
  #status { display:none; }
}
</style>
</head><body>

<canvas id="gl"></canvas>
<canvas id="hud"></canvas>

<div class="panel" id="brand">
  <div class="word">d<b>w</b>ago</div>
  <div class="sub" id="brandsub">—</div>
</div>

<div class="panel" id="searchp">
  <input id="q" placeholder="search the galaxy…  ( / )" autocomplete="off" spellcheck="false">
  <div id="results"></div>
</div>

<div class="panel" id="ctrl">
  <div class="plabel">spectral view</div>
  <div class="row">
    <select id="mode" title="recolor the stars (keys 1-6)">
      <option value="natural">1 natural</option>
      <option value="community">2 community</option>
      <option value="hotspot">3 hotspot</option>
      <option value="age">4 age</option>
      <option value="bus">5 bus factor</option>
      <option value="owner">6 owner</option>
    </select>
    <button id="bblast" title="show the 3-hop impact web of the selected star">blast</button>
    <button id="bpause" title="pause the drift">||</button>
  </div>
  <div class="plabel">history</div>
  <input type="range" id="time" min="0" max="100" value="100">
  <div id="timelabel">showing everything</div>
</div>

<div class="panel" id="detail">
  <button class="close" id="dclose">×</button>
  <h3 id="dname">—</h3>
  <div class="loc" id="dloc"></div>
  <div id="dpills"></div>
  <div class="plabel" style="margin-top:10px">linked <span style="color:var(--amber)">·</span> = co-change</div>
  <div id="dneigh"></div>
</div>

<div class="panel" id="status">—</div>
<div id="tip"></div>

<script id="payload" type="application/json">__DATA__</script>
<script>
'use strict';
/* ── data ─────────────────────────────────────────────────────────────────── */
const D = JSON.parse(document.getElementById('payload').textContent);
const dec = (b, T) => { const s = atob(b), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
const N = D.n, MISC = D.miscId;
const comm = dec(D.community, Int32Array), gcm = dec(D.gcomm, Int32Array);
const hot = dec(D.hotspot, Float32Array), ageArr = dec(D.age, Int32Array);
const busArr = dec(D.bus, Int32Array), ownIdx = dec(D.ownerIdx, Int32Array);
const lineNo = dec(D.line, Int32Array), degArr = dec(D.deg, Int32Array);
const srcA = dec(D.src, Int32Array), dstA = dec(D.dst, Int32Array);
const chanA = dec(D.chan, Int32Array);
const P = dec(D.pos, Float32Array);
const NAT = dec(D.natural, Uint8Array);
const DUST = dec(D.dust, Float32Array);
const GLOW = dec(D.glow, Float32Array);
const BG = dec(D.bg, Float32Array), BGS = dec(D.bgSize, Float32Array),
      BGP = dec(D.bgPhase, Float32Array), BGC = dec(D.bgCol, Uint8Array);
const GAL = dec(D.gal, Float32Array), GALS = dec(D.galSize, Float32Array),
      GALA = dec(D.galAngle, Float32Array), GALE = dec(D.galEll, Float32Array),
      GALC = dec(D.galCol, Uint8Array);
const ROUTES = dec(D.routes, Int32Array);
const FIL = dec(D.fil, Float32Array), FILC = dec(D.filCol, Uint8Array),
      FILS = dec(D.filSize, Float32Array);
const NEB = dec(D.neb, Float32Array), NEBC = dec(D.nebCol, Uint8Array),
      NEBS = dec(D.nebSize, Float32Array), NEBA = dec(D.nebAlpha, Float32Array);
const labels = D.labels, kinds = D.kinds, paths = D.paths;
const E = srcA.length;
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* adjacency */
const off = new Int32Array(N + 1);
{ const d = new Int32Array(N);
  for (let i = 0; i < E; i++) { d[srcA[i]]++; d[dstA[i]]++; }
  for (let i = 0; i < N; i++) off[i+1] = off[i] + d[i]; }
const adj = new Int32Array(off[N]), adjCh = new Int8Array(off[N]);
{ const cur = Int32Array.from(off.subarray(0, N));
  for (let i = 0; i < E; i++) {
    adj[cur[srcA[i]]] = dstA[i]; adjCh[cur[srcA[i]]++] = chanA[i];
    adj[cur[dstA[i]]] = srcA[i]; adjCh[cur[dstA[i]]++] = chanA[i]; } }

/* ── spectral palettes ────────────────────────────────────────────────────── */
const HUES = [[110,150,255],[150,120,255],[255,120,180],[110,220,205],[255,200,110],
              [130,230,140],[255,120,120],[110,200,255],[210,230,120],[240,140,255]];
const hueOf = i => HUES[((i % HUES.length) + HUES.length) % HUES.length];
const KIND_HUE = {}; { let k = 0; for (const s of kinds) if (!(s in KIND_HUE)) KIND_HUE[s] = hueOf(k++); }
let maxHot = 0; for (const h of hot) if (h > maxHot) maxHot = h;
let minAge = Infinity, maxAge = 0;
for (const a of ageArr) if (a > 0) { if (a < minAge) minAge = a; if (a > maxAge) maxAge = a; }
if (!isFinite(minAge)) { minAge = 0; maxAge = 1; }
const ramp = t => { t = Math.max(0, Math.min(1, t));
  return [140 + 115*t, 170 - 60*t, 255 - 175*t]; };   // cool blue -> hot ember

let mode = 'natural';
function starColor(i) {
  if (mode === 'natural')  return [NAT[i*3], NAT[i*3+1], NAT[i*3+2]];
  if (mode === 'community') return gcm[i] === MISC ? [120,124,134] : hueOf(gcm[i]);
  if (mode === 'kind')     return KIND_HUE[kinds[i]] || [150,150,160];
  if (mode === 'hotspot')  return maxHot > 0 ? ramp(Math.sqrt(hot[i]/maxHot)) : [120,124,134];
  if (mode === 'age')      return ageArr[i] > 0 ? ramp((ageArr[i]-minAge)/Math.max(1,maxAge-minAge)) : [110,114,124];
  if (mode === 'bus')      return busArr[i] === 1 ? [255,110,100] : busArr[i] === 2 ? [255,200,110] : [120,210,160];
  return hueOf(ownIdx[i]);
}

const visible = new Uint8Array(N).fill(1);

/* ── WebGL ────────────────────────────────────────────────────────────────── */
const cv = document.getElementById('gl');
const hud = document.getElementById('hud');
const hctx = hud.getContext('2d');
let W = 0, H = 0, DPR = 1;
const gl = cv.getContext('webgl', { antialias: false, alpha: false,
                                    premultipliedAlpha: false });
if (!gl) document.getElementById('status').textContent = 'WebGL unavailable';

const VS = `
precision highp float;
attribute vec3 aPos; attribute vec3 aCol;
attribute float aSize, aPhase, aKind, aAux;
uniform float uYaw, uPitch, uScale, uAspect, uDpr, uTime;
varying vec3 vCol; varying float vKind, vAux, vPhase, vDepth, vTw;
void main() {
  float cy = cos(uYaw), sy = sin(uYaw);
  vec3 p = vec3(cy*aPos.x + sy*aPos.z, aPos.y, -sy*aPos.x + cy*aPos.z);
  float cx = cos(uPitch), sx = sin(uPitch);
  p = vec3(p.x, cx*p.y - sx*p.z, sx*p.y + cx*p.z);
  p *= uScale;
  float zc = p.z + 1250.0;
  float f = 820.0 / max(zc, 40.0);
  gl_Position = vec4(p.x * f / (430.0 * uAspect), p.y * f / 430.0,
                     p.z / 6000.0, 1.0);
  vDepth = clamp((zc - 650.0) / 1400.0, 0.0, 1.0);
  vCol = aCol / 255.0; vKind = aKind; vAux = aAux; vPhase = aPhase;
  vTw = 0.90 + 0.10 * sin(uTime * (1.1 + fract(aPhase) * 1.3) + aPhase * 7.0);
  float sz = aSize;
  if (aKind < 0.5) sz *= 1.0 + 0.07 * sin(uTime * 1.7 + aPhase * 3.0);   // stars breathe
  float px = sz * f * uDpr * 0.62;
  /* Zoom must not inflate stars into blobs: a star stays a pinpoint however
     close you get — real cameras resolve brightness, not diameter. Haze,
     nebula and dust (kind 1,2) are genuinely extended objects and may scale. */
  if (aKind < 0.5)      px = min(px, 11.0 * uDpr);   // data stars
  else if (aKind > 2.5 && aKind < 3.5) px = min(px, 6.0 * uDpr);   // filler/field
  else if (aKind > 3.5 && aKind < 4.5) px = min(px, 26.0 * uDpr);  // galaxies
  else if (aKind > 5.5) px = min(px, 9.0 * uDpr);    // pulses
  gl_PointSize = px;
}`;
const FS = `
precision mediump float;
uniform float uTime;
varying vec3 vCol; varying float vKind, vAux, vPhase, vDepth, vTw;
void main() {
  vec2 q = gl_PointCoord - vec2(0.5);
  float r2 = dot(q, q) * 4.0;
  if (r2 > 1.0) discard;
  float fog = 1.0 - vDepth * 0.55;
  if (vKind < 0.5) {                       /* data star: gaussian + hot core */
    if (vAux < 0.5) discard;               /* aAux carries visibility */
    float g = exp(-r2 * 5.0);
    float core = exp(-r2 * 26.0);
    vec3 c = vCol * g + mix(vCol, vec3(1.0), 0.55) * core * 0.55;
    gl_FragColor = vec4(c, (g * 0.72 + core * 0.8) * fog);
  } else if (vKind < 1.5) {                /* haze */
    float g = exp(-r2 * 2.2);
    gl_FragColor = vec4(vCol * g, g * vAux * fog);
  } else if (vKind < 2.5) {                /* dust: subtractive */
    float g = exp(-r2 * 2.6);
    gl_FragColor = vec4(0.0, 0.0, 0.0, g * vAux);
  } else if (vKind < 3.5) {                /* background star, twinkling */
    float g = exp(-r2 * 6.0);
    float core = exp(-r2 * 30.0);
    gl_FragColor = vec4(vCol * g + vec3(1.0) * core * 0.35, (g * 0.68 + core * 0.7) * vTw);
  } else if (vKind < 4.5) {                /* distant galaxy smudge */
    float ca = cos(vPhase), sa = sin(vPhase);
    vec2 e = vec2(ca*q.x + sa*q.y, (-sa*q.x + ca*q.y) / max(vAux, 0.2));
    float g = exp(-dot(e, e) * 7.0);
    gl_FragColor = vec4(vCol * g, g * 0.35);
  } else if (vKind < 5.5) {                /* diffraction spikes, 45deg cross */
    vec2 d = vec2(q.x + q.y, q.x - q.y) * 1.4142;
    float ray = exp(-abs(d.x) * 34.0) + exp(-abs(d.y) * 34.0);
    float env = exp(-r2 * 1.4);
    gl_FragColor = vec4(vCol, ray * env * 0.34 * fog);
  } else {                                 /* signal pulse + trail ghost */
    float g = exp(-r2 * 6.0);
    gl_FragColor = vec4(vCol * g + vec3(1.0) * exp(-r2 * 26.0) * 0.4,
                        g * 0.7 * vAux * fog);
  }
}`;

function compile(kind, srcTxt) {
  const h = gl.createShader(kind);
  gl.shaderSource(h, srcTxt); gl.compileShader(h);
  if (!gl.getShaderParameter(h, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(h));
  return h;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
gl.useProgram(prog);
const U = {}; for (const u of ['uYaw','uPitch','uScale','uAspect','uDpr','uTime'])
  U[u] = gl.getUniformLocation(prog, u);
const A = {}; for (const a of ['aPos','aCol','aSize','aPhase','aKind','aAux'])
  { A[a] = gl.getAttribLocation(prog, a); gl.enableVertexAttribArray(A[a]); }

/* one interleaved batch per pass: [x y z r g b size phase kind aux] */
const STRIDE = 10;
function makeBatch(count) {
  return { n: 0, cap: count, data: new Float32Array(count * STRIDE),
           buf: gl.createBuffer(), dirty: true };
}
function push(b, x, y, z, r, g, bl, size, phase, kind, aux) {
  const o = b.n * STRIDE, d = b.data;
  d[o]=x; d[o+1]=y; d[o+2]=z; d[o+3]=r; d[o+4]=g; d[o+5]=bl;
  d[o+6]=size; d[o+7]=phase; d[o+8]=kind; d[o+9]=aux; b.n++;
}
function drawBatch(b, blend) {
  if (!b.n) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, b.buf);
  if (b.dirty) { gl.bufferData(gl.ARRAY_BUFFER, b.data.subarray(0, b.n*STRIDE), gl.DYNAMIC_DRAW); b.dirty = false; }
  const F = 4;
  gl.vertexAttribPointer(A.aPos,   3, gl.FLOAT, false, STRIDE*F, 0);
  gl.vertexAttribPointer(A.aCol,   3, gl.FLOAT, false, STRIDE*F, 3*F);
  gl.vertexAttribPointer(A.aSize,  1, gl.FLOAT, false, STRIDE*F, 6*F);
  gl.vertexAttribPointer(A.aPhase, 1, gl.FLOAT, false, STRIDE*F, 7*F);
  gl.vertexAttribPointer(A.aKind,  1, gl.FLOAT, false, STRIDE*F, 8*F);
  gl.vertexAttribPointer(A.aAux,   1, gl.FLOAT, false, STRIDE*F, 9*F);
  if (blend === 'add') gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  else if (blend === 'dust') gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA);
  else gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.drawArrays(gl.POINTS, 0, b.n);
}

/* ── build batches ────────────────────────────────────────────────────────── */
const nebula = makeBatch(NEBS.length);
for (let i = 0; i < NEBS.length; i++)
  push(nebula, NEB[i*3], NEB[i*3+1], NEB[i*3+2],
       NEBC[i*3], NEBC[i*3+1], NEBC[i*3+2], NEBS[i], 0, 1, NEBA[i]);

const backdrop = makeBatch(BG.length/3 + GAL.length);
for (let i = 0; i < BG.length/3; i++)
  push(backdrop, BG[i*3], BG[i*3+1], BG[i*3+2], BGC[i*3], BGC[i*3+1], BGC[i*3+2],
       BGS[i], BGP[i], 3, 1);
for (let i = 0; i < GALS.length; i++)
  push(backdrop, GAL[i*3], GAL[i*3+1], GAL[i*3+2], GALC[i*3], GALC[i*3+1], GALC[i*3+2],
       GALS[i], GALA[i], 4, GALE[i]);

/* unresolved-star filler: draws under the data stars, additive */
const filler = makeBatch(FIL.length/3);
for (let i = 0; i < FIL.length/3; i++)
  push(filler, FIL[i*3], FIL[i*3+1], FIL[i*3+2],
       FILC[i*3], FILC[i*3+1], FILC[i*3+2], FILS[i], (i*1.7)%6.283, 3, 1);

const haze = makeBatch(200);
{ /* core glow: layered warm sprites + community-cluster nebulosity */
  const layers = [[120, .50],[250, .28],[460, .12],[720, .05]];
  for (const [s, a] of layers) push(haze, 0, 0, 0, 255, 205, 150, s, 0, 1, a);
  /* disc body: faint warm-cream membrane across the whole plane */
  for (let k = 0; k < 48; k++) {
    const a = k * 2.399963, r = 50 + 230 * Math.sqrt((k + 1) / 48);
    push(haze, Math.cos(a) * r, 0, Math.sin(a) * r, 235, 220, 200, 130, 0, 1, 0.05);
  }
  const agg = new Map();
  for (let i = 0; i < N; i++) {
    if (gcm[i] === MISC) continue;
    let m = agg.get(gcm[i]); if (!m) agg.set(gcm[i], m = {x:0,y:0,z:0,n:0,r:0,g:0,b:0});
    m.x+=P[i*3]; m.y+=P[i*3+1]; m.z+=P[i*3+2]; m.n++;
    m.r+=NAT[i*3]; m.g+=NAT[i*3+1]; m.b+=NAT[i*3+2];
  }
  for (const m of agg.values()) {
    if (m.n < 12) continue;
    push(haze, m.x/m.n, m.y/m.n, m.z/m.n, m.r/m.n, m.g/m.n, m.b/m.n,
         30 + 5*Math.sqrt(m.n), 0, 1, 0.10);
  }
}

const armglow = makeBatch(GLOW.length/4);
for (let i = 0; i < GLOW.length/4; i++)
  push(armglow, GLOW[i*4], GLOW[i*4+1], GLOW[i*4+2], 118, 152, 232,
       GLOW[i*4+3], 0, 1, 0.115);

const dust = makeBatch(DUST.length/4);
for (let i = 0; i < DUST.length/4; i++)
  push(dust, DUST[i*4], DUST[i*4+1], DUST[i*4+2], 0,0,0, DUST[i*4+3], 0, 2, 0.50);

const stars = makeBatch(N);
function starSize(i) { return gcm[i]===MISC
  ? 1.2 + Math.min(3, Math.sqrt(degArr[i]) * .5)
  : 2.1 + Math.min(9, Math.sqrt(degArr[i]) * 1.15); }
function rebuildStars() {
  stars.n = 0;
  for (let i = 0; i < N; i++) {
    let [r, g, b] = starColor(i);
    if (gcm[i] === MISC && mode === 'natural') { r *= .55; g *= .55; b *= .55; }
    push(stars, P[i*3], P[i*3+1], P[i*3+2], r, g, b,
         starSize(i), (i * 2.4) % 6.283, 0, visible[i]);
  }
  stars.dirty = true;
}
rebuildStars();

/* Diffraction spikes removed: with 40 of them flaring at once the scene read
   as glitter, not a photograph. The hot cores were doing the same at smaller
   scale — both are toned down in the shader instead. */
function rebuildSpikes() {}

/* The neural life from the tissue view, living inside the galaxy: signals
   travel real edges with comet trails, and on arrival chain onward through the
   destination star's own links — visible cascades of activity. */
const NPULSE = reduced ? 0 : 150;
const comets = makeBatch(Math.max(NPULSE * 3, 1));
const pulses = [];
for (let p = 0; p < NPULSE; p++) {
  const e = ROUTES[p % ROUTES.length];
  pulses.push({ a: srcA[e], b: dstA[e], t: (p * 0.37) % 1,
                v: 0.004 + (p % 5) * 0.0014 });
}
function nextHop(from) {
  const n = off[from+1] - off[from];
  if (!n) return -1;
  return adj[off[from] + ((Math.random() * n) | 0)];
}

/* ── camera: inertial, smoothed — responsiveness is the product ──────────── */
let yaw = 0.6, pitch = 0.96, scale = 1.75;
let yawT = yaw, pitchT = pitch, scaleT = 1.85;
let yawVel = 0;
let running = !reduced, dragging = false;

function frame(ts, dt) {
  /* smoothing: every input edits a target; the camera eases toward it.
     Zoom/rotate therefore never snaps or stalls — the top complaint about
     the old renderer was exactly that seam. */
  if (!dragging && running) yawT += dt * 0.0000105;
  yaw += (yawT - yaw) * 0.16 + yawVel;
  yawVel *= 0.90;
  pitch += (pitchT - pitch) * 0.16;
  scale += (scaleT - scale) * 0.18;

  gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0.008, 0.011, 0.023, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.uniform1f(U.uYaw, yaw); gl.uniform1f(U.uPitch, pitch);
  gl.uniform1f(U.uScale, scale); gl.uniform1f(U.uAspect, W / H);
  gl.uniform1f(U.uDpr, DPR); gl.uniform1f(U.uTime, ts / 1000);

  drawBatch(nebula, 'add');
  drawBatch(backdrop, 'add');
  drawBatch(armglow, 'add');
  drawBatch(filler, 'add');
  drawBatch(haze, 'add');
  drawBatch(dust, 'dust');
  drawBatch(stars, 'add');

  if (running && pulses.length) {
    comets.n = 0;
    for (const p of pulses) {
      p.t += p.v * (dt / 16.7);
      if (p.t >= 1) {
        /* arrival: chain onward through the destination's own links (70%),
           otherwise respawn on a fresh route — the cascades read as thought
           propagating through the galaxy, exactly the tissue-view behaviour */
        const hop = Math.random() < 0.7 ? nextHop(p.b) : -1;
        if (hop >= 0 && visible[hop]) { p.a = p.b; p.b = hop; }
        else {
          const e = ROUTES[(Math.random() * ROUTES.length) | 0];
          p.a = srcA[e]; p.b = dstA[e];
        }
        p.t = 0;
      }
      if (!visible[p.a] || !visible[p.b]) continue;
      /* comet head + two fading trail ghosts along the segment */
      for (let g2 = 0; g2 < 3; g2++) {
        const t = Math.max(0, p.t - g2 * 0.06), u = 1 - t;
        push(comets,
             u*P[p.a*3]+t*P[p.b*3], u*P[p.a*3+1]+t*P[p.b*3+1], u*P[p.a*3+2]+t*P[p.b*3+2],
             g2 === 0 ? 225 : 160, g2 === 0 ? 235 : 195, 255,
             3.0 - g2 * 0.9, 0, 6, 1 - g2 * 0.3);
      }
    }
    comets.dirty = true;
    drawBatch(comets, 'add');
  }

  drawHud(ts);
}

/* ── projection on the CPU (picking + HUD overlay) ────────────────────────── */
function project(i, out) {
  const x = P[i*3], y = P[i*3+1], z = P[i*3+2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  let x1 = cy*x + sy*z, z1 = -sy*x + cy*z;
  const cx = Math.cos(pitch), sx = Math.sin(pitch);
  let y1 = cx*y - sx*z1, z2 = sx*y + cx*z1;
  x1 *= scale; y1 *= scale; z2 *= scale;
  const zc = z2 + 1250;
  if (zc < 60) return false;
  const f = 820 / zc;
  out[0] = (x1 * f / (430 * (W/H))) * (W/2) + W/2;
  out[1] = H/2 - (y1 * f / 430) * (H/2);
  out[2] = zc;
  return true;
}
const _pt = [0,0,0];
function pick(mx, my) {
  let best = -1, bd = 18 * 18;
  for (let i = 0; i < N; i++) {
    if (!visible[i]) continue;
    if (!project(i, _pt)) continue;
    const dx = _pt[0]-mx, dy = _pt[1]-my, d = dx*dx + dy*dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

/* ── HUD overlay: reticles + constellation lines (edges on demand) ────────── */
let hovered = -1, selected = -1, blastOn = false, blastSet = null;
function drawHud(ts) {
  hctx.clearRect(0, 0, W, H);
  if (selected >= 0 && project(selected, _pt)) {
    const [px, py] = _pt;
    /* constellation: this star's links, mission-annotation style */
    const p2 = [0,0,0];
    hctx.lineWidth = 1;
    let shown = 0;
    for (let k = off[selected]; k < off[selected+1] && shown < 60; k++) {
      const j = adj[k];
      if (!visible[j] || !project(j, p2)) continue;
      hctx.strokeStyle = adjCh[k] ? 'rgba(255,190,114,.55)' : 'rgba(142,198,255,.38)';
      if (adjCh[k]) hctx.setLineDash([4,4]); else hctx.setLineDash([]);
      hctx.beginPath(); hctx.moveTo(px, py); hctx.lineTo(p2[0], p2[1]); hctx.stroke();
      hctx.fillStyle = 'rgba(200,220,255,.8)';
      hctx.fillRect(p2[0]-1, p2[1]-1, 2, 2);
      shown++;
    }
    hctx.setLineDash([]);
    /* blast web: second ring, fainter */
    if (blastOn && blastSet) {
      hctx.strokeStyle = 'rgba(142,198,255,.14)';
      for (const [a, b] of blastSet.edges) {
        if (!project(a, _pt) ) continue;
        const pa0 = _pt[0], pa1 = _pt[1];
        if (!project(b, p2)) continue;
        hctx.beginPath(); hctx.moveTo(pa0, pa1); hctx.lineTo(p2[0], p2[1]); hctx.stroke();
      }
      project(selected, _pt);
    }
    /* reticle */
    const [rx, ry] = _pt;
    const R = 13 + (running ? 1.6*Math.sin(ts/300) : 0);
    hctx.strokeStyle = 'rgba(142,198,255,.9)';
    hctx.lineWidth = 1.2;
    for (const [a0, a1] of [[0.15,1.42],[1.72,2.99],[3.29,4.56],[4.86,6.13]]) {
      hctx.beginPath(); hctx.arc(rx, ry, R, a0, a1); hctx.stroke();
    }
    hctx.font = '11px ' + getComputedStyle(document.body).getPropertyValue('--mono');
    hctx.fillStyle = 'rgba(220,230,245,.9)';
    hctx.fillText(labels[selected].slice(0, 38), rx + R + 7, ry + 4);
  }
  if (hovered >= 0 && hovered !== selected && project(hovered, _pt)) {
    hctx.strokeStyle = 'rgba(220,230,245,.6)';
    hctx.lineWidth = 1;
    hctx.beginPath(); hctx.arc(_pt[0], _pt[1], 9, 0, 7); hctx.stroke();
  }
}

/* ── interaction ──────────────────────────────────────────────────────────── */
const ptrs = new Map(); let pinchD = 0, moved = false, lx = 0, ly = 0;
cv.addEventListener('pointerdown', e => {
  ptrs.set(e.pointerId, [e.clientX, e.clientY]);
  if (ptrs.size === 2) { const [a,b] = [...ptrs.values()];
    pinchD = Math.hypot(a[0]-b[0], a[1]-b[1]); dragging = false; return; }
  dragging = true; moved = false; lx = e.clientX; ly = e.clientY;
  try { cv.setPointerCapture(e.pointerId); } catch (_) {}
});
addEventListener('pointerup', e => {
  ptrs.delete(e.pointerId);
  if (dragging && !moved) {
    const i = pick(e.clientX, e.clientY);
    if (i >= 0) select(i);
    else { selected = -1; blastSet = null; closeDetail(); if (!blastOn) baseStatus(); }
  }
  dragging = false;
});
addEventListener('pointercancel', e => { ptrs.delete(e.pointerId); dragging = false; });
addEventListener('pointermove', e => {
  if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, [e.clientX, e.clientY]);
  if (ptrs.size === 2) {
    const [a,b] = [...ptrs.values()];
    const d = Math.hypot(a[0]-b[0], a[1]-b[1]);
    if (pinchD > 0) scaleT = Math.max(.3, Math.min(6, scaleT * d / pinchD));
    pinchD = d; return;
  }
  if (dragging && e.buttons === 0) { dragging = false; return; }
  if (dragging) {
    const dx = e.clientX - lx, dy = e.clientY - ly;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    yawT += dx * 0.0052; yaw += dx * 0.0052;          // direct + target: zero lag
    pitchT = Math.max(-1.35, Math.min(1.35, pitchT + dy * 0.0052));
    lx = e.clientX; ly = e.clientY;
  } else {
    if (e.target !== cv) { if (hovered !== -1) { hovered = -1; tip(-1,0,0); } return; }
    hoverAt(e.clientX, e.clientY);
  }
});
let hoverPending = 0;
function hoverAt(mx, my) {
  /* picking walks 14k nodes; throttle to one per frame */
  if (hoverPending) return;
  hoverPending = requestAnimationFrame(() => {
    hoverPending = 0;
    const i = pick(mx, my);
    if (i !== hovered) { hovered = i; tip(i, mx, my); }
    else if (i >= 0) tip(i, mx, my);
  });
}
cv.addEventListener('wheel', e => {
  e.preventDefault();
  scaleT = Math.max(.3, Math.min(6, scaleT * Math.exp(-e.deltaY * 0.0016)));
}, { passive: false });

const tipEl = document.getElementById('tip');
function tip(i, mx, my) {
  if (i < 0) { tipEl.style.display = 'none'; return; }
  tipEl.innerHTML = esc(labels[i]) + '<small>' + esc((paths[i]||'').split('/').pop()) + '</small>';
  tipEl.style.display = 'block';
  tipEl.style.left = Math.min(mx + 14, W - 354) + 'px';
  tipEl.style.top = (my + 16) + 'px';
}
const esc = s => String(s).replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]));

/* fly the camera so star i faces us */
function flyTo(i) {
  const x = P[i*3], z = P[i*3+2];
  let bestA = yawT, bestZ = 1e9;
  for (let k = 0; k < 96; k++) {
    const a = k / 96 * 6.28318;
    const z1 = -Math.sin(a)*x + Math.cos(a)*z;
    if (z1 < bestZ) { bestZ = z1; bestA = a; }
  }
  // unwrap to the nearest turn so the camera takes the short way round
  const cur = yawT % 6.28318;
  let delta = bestA - cur;
  if (delta > 3.14159) delta -= 6.28318;
  if (delta < -3.14159) delta += 6.28318;
  yawT += delta;
  pitchT = 0.85;
  scaleT = Math.max(scaleT, 1.8);
}

/* ── selection / detail ───────────────────────────────────────────────────── */
const detail = document.getElementById('detail');
function select(i) {
  selected = i;
  document.getElementById('dname').textContent = labels[i];
  document.getElementById('dloc').textContent = (paths[i]||'—') + (lineNo[i] ? ':' + lineNo[i] : '');
  let pills = `<span class="pill">${esc(kinds[i]||'?')}</span>` +
              `<span class="pill">cluster ${comm[i]}</span>` +
              `<span class="pill">${off[i+1]-off[i]} links</span>`;
  if (busArr[i] === 1) pills += `<span class="pill risk">bus factor 1</span>`;
  if (hot[i] > 0) pills += `<span class="pill hot">hotspot ${hot[i].toFixed(0)}</span>`;
  document.getElementById('dpills').innerHTML = pills;
  const nb = [];
  for (let k = off[i]; k < off[i+1]; k++) nb.push([adj[k], adjCh[k]]);
  nb.sort((a,b) => (off[b[0]+1]-off[b[0]]) - (off[a[0]+1]-off[a[0]]));
  document.getElementById('dneigh').innerHTML = nb.slice(0, 30).map(([j, ch]) =>
    `<div class="nb" data-i="${j}"><span class="tag${ch?' tmp':''}">${ch?'·':'—'}</span>` +
    `${esc(labels[j])}</div>`).join('');
  document.querySelectorAll('#dneigh .nb').forEach(el =>
    el.onclick = () => { const j = +el.dataset.i; select(j); flyTo(j); });
  detail.classList.add('open');
  if (blastOn) blast(i);
}
function closeDetail() { detail.classList.remove('open'); }
document.getElementById('dclose').onclick = () => {
  selected = -1; blastSet = null; closeDetail(); if (!blastOn) baseStatus(); };

function blast(root) {
  const h = new Float32Array(N); h[root] = 1;
  const edges = [];
  let frontier = [root];
  const decay = [1, .55, .3, .16];
  for (let hop = 0; hop < 3 && frontier.length; hop++) {
    const nxt = [];
    for (const u of frontier)
      for (let k = off[u]; k < off[u+1]; k++) {
        const v = adj[k];
        if (!visible[v]) continue;
        const w = decay[hop+1] * (adjCh[k] ? .8 : 1);
        if (w > h[v]) {
          if (!h[v]) edges.push([u, v]);
          h[v] = w; nxt.push(v);
        } }
    frontier = nxt;
    if (frontier.length > 20000) break;
  }
  let cnt = 0; for (const x of h) if (x > 0) cnt++;
  blastSet = { mask: h, edges: edges.slice(0, 900) };
  status(`blast: <b>${cnt.toLocaleString()}</b> stars within 3 hops`);
}

/* ── search ───────────────────────────────────────────────────────────────── */
const index = new Map();
const addTok = (t, i) => { if (t.length < 2) return;
  let a = index.get(t); if (!a) index.set(t, a = []);
  if (a.length < 400 && a[a.length-1] !== i) a.push(i); };
for (let i = 0; i < N; i++) {
  const s = (labels[i] + ' ' + (paths[i]||'')).toLowerCase();
  for (const m of s.split(/[^a-z0-9]+/)) if (m) addTok(m, i);
}
const qEl = document.getElementById('q'), resEl = document.getElementById('results');
let selIdx = -1;
qEl.addEventListener('input', () => {
  selIdx = -1;
  const q = qEl.value.trim().toLowerCase();
  if (q.length < 2) { resEl.classList.remove('open'); resEl.innerHTML=''; return; }
  const toks = q.split(/[^a-z0-9]+/).filter(Boolean);
  const score = new Map();
  for (const t of toks)
    for (const [key, arr] of index) {
      if (!key.startsWith(t)) continue;
      const w = key === t ? 2 : 1/(1+key.length-t.length);
      for (const i of arr) score.set(i, (score.get(i)||0) + w); }
  const top = [...score.entries()].sort((a,b) => b[1]-a[1]).slice(0, 30);
  resEl.innerHTML = top.map(([i]) =>
    `<div class="hit" data-i="${i}">${esc(labels[i])}<small>${esc((paths[i]||'').split('/').slice(-2).join('/'))}</small></div>`
  ).join('') || '<div class="hit" style="pointer-events:none;color:var(--faint)">no match — try a shorter prefix</div>';
  resEl.classList.add('open');
  resEl.querySelectorAll('.hit[data-i]').forEach(el =>
    el.onclick = () => { const i = +el.dataset.i;
      resEl.classList.remove('open'); qEl.blur();
      select(i); flyTo(i); });
});
addEventListener('keydown', e => {
  if (e.key === '/' && document.activeElement !== qEl) { e.preventDefault(); qEl.focus(); return; }
  if (e.key === 'Escape') {
    resEl.classList.remove('open'); selIdx = -1;
    selected = -1; blastSet = null; closeDetail(); qEl.blur();
    return;
  }
  if (document.activeElement === qEl && resEl.classList.contains('open')) {
    const hits = [...resEl.querySelectorAll('.hit[data-i]')];
    if (!hits.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      selIdx = (selIdx + (e.key === 'ArrowDown' ? 1 : -1) + hits.length) % hits.length;
      hits.forEach((h, i) => h.classList.toggle('sel', i === selIdx));
      hits[selIdx].scrollIntoView({ block: 'nearest' });
      return;
    }
    if (e.key === 'Enter' && selIdx >= 0) { e.preventDefault(); hits[selIdx].click(); return; }
  }
  if (document.activeElement !== qEl && e.key >= '1' && e.key <= '6') {
    const m = document.getElementById('mode');
    m.selectedIndex = +e.key - 1;
    m.dispatchEvent(new Event('change'));
  }
});

/* ── controls ─────────────────────────────────────────────────────────────── */
const statusEl = document.getElementById('status');
const status = h => statusEl.innerHTML = h;
const baseStatus = () =>
  status(`<b>${N.toLocaleString()}</b> stars · <b>${E.toLocaleString()}</b> links · offline`);
document.getElementById('mode').onchange = e => {
  mode = e.target.value; rebuildStars(); rebuildSpikes(); };
const bpause = document.getElementById('bpause');
bpause.onclick = () => {
  running = !running;
  bpause.classList.toggle('on', !running);
  bpause.textContent = running ? '||' : '>';
};
document.getElementById('bblast').onclick = e => {
  blastOn = !blastOn; e.target.classList.toggle('on', blastOn);
  if (blastOn && selected >= 0) blast(selected);
  else if (blastOn) status('blast armed — click a star to see its 3-hop web');
  else { blastSet = null; baseStatus(); } };

const timeEl = document.getElementById('time');
let scrubTimer = 0;
timeEl.oninput = () => {
  const f = +timeEl.value / 100;
  if (f >= 1) { visible.fill(1);
    document.getElementById('timelabel').textContent = 'showing everything'; }
  else {
    const cut = minAge + (maxAge - minAge) * f;
    let shown = 0;
    for (let i = 0; i < N; i++) {
      visible[i] = (ageArr[i] === 0 || ageArr[i] <= cut) ? 1 : 0;
      shown += visible[i]; }
    const d = new Date(cut * 1000);
    document.getElementById('timelabel').textContent =
      d.toISOString().slice(0,10) + ' — ' + shown.toLocaleString() + ' stars';
  }
  clearTimeout(scrubTimer);
  scrubTimer = setTimeout(() => {
    /* visibility rides in aAux — update one interleaved field, no rebuild */
    for (let i = 0; i < N; i++) stars.data[i*STRIDE + 9] = visible[i];
    stars.dirty = true;
    rebuildSpikes();
    if (blastOn && selected >= 0) blast(selected);
  }, 90);
};

/* ── boot ─────────────────────────────────────────────────────────────────── */
function size() {
  DPR = Math.min(devicePixelRatio || 1, 2);
  W = innerWidth; H = innerHeight;
  cv.width = W * DPR; cv.height = H * DPR;
  hud.width = W * DPR; hud.height = H * DPR;
  hctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
size();
addEventListener('resize', size);
document.getElementById('brandsub').textContent = __TITLE_JS__;
baseStatus();
if (reduced) { running = false; bpause.classList.add('on'); bpause.textContent = '>'; }

let lastT = 0;
function loop(ts) {
  const dt = Math.min(50, lastT ? ts - lastT : 16); lastT = ts;
  try { frame(ts, dt); }
  catch (err) { console.error(err); }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
</script>
</body></html>
"""
