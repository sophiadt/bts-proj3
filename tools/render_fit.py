"""Pictures of a fitted subject: before/after, scan match, inside, animation.

Reads template_export/subjects/<id>/ (written by pipeline/fit_subject.py, and
anim_*.npz by rig/sim_check.py and rig/ui_smoke_test.py) and writes PNGs and
GIFs to template_export/subjects/<id>/views/. Matplotlib painter's-algorithm
shading, no Maya needed.

    mayapy tools/render_fit.py capture_01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core import geometry as geo                       # noqa: E402
from pipeline import fixture as fx                     # noqa: E402

BG = "#f7f8f9"
SKIN_RGB = np.array([0.80, 0.66, 0.58])
COLOUR = {"bone": "#8a8170", "tissue": "#b03a3a", "cloth": "#c98b62"}


def project(P, view):
    if view == "front":       # viewer at -Z: the subject's right (+X) on the viewer's left
        return np.c_[-P[:, 0], P[:, 1]], P[:, 2]
    return np.c_[-P[:, 2], P[:, 1]], -P[:, 0]     # from the subject's right side, face to the right


def draw(ax, P, F, view, rgb=SKIN_RGB, values=None, vmax=None, cmap="viridis"):
    xy, depth = project(P, view)
    n = geo.face_normals(P, F)
    light = np.array([-0.3, 0.4, -1.0]) if view == "front" else np.array([1.0, 0.4, -0.3])
    shade = 0.35 + 0.65 * np.abs(n @ (light / np.linalg.norm(light)))
    order = np.argsort(-depth[F].mean(axis=1))
    if values is not None:
        base = plt.get_cmap(cmap)(np.clip(values[F].mean(axis=1) / max(vmax, 1e-9), 0, 1))[:, :3]
    else:
        base = np.tile(rgb, (len(F), 1))
    cols = np.clip(base * shade[:, None], 0, 1)
    ax.add_collection(PolyCollection(xy[F[order]], facecolors=cols[order], edgecolors="none", antialiased=False))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(xy[:, 0].min() - 5, xy[:, 0].max() + 5)
    ax.set_ylim(xy[:, 1].min() - 5, xy[:, 1].max() + 5)


def draw_parts(ax, parts, view):
    """Several meshes depth-sorted together, so whatever sits in front shows."""
    polys, cols, depths = [], [], []
    light = np.array([-0.3, 0.4, -1.0]) if view == "front" else np.array([1.0, 0.4, -0.3])
    light = light / np.linalg.norm(light)
    for P, F, rgb in parts:
        xy, depth = project(P, view)
        shade = 0.35 + 0.65 * np.abs(geo.face_normals(P, F) @ light)
        polys.append(xy[F]); depths.append(depth[F].mean(axis=1))
        cols.append(np.clip(np.tile(rgb, (len(F), 1)) * shade[:, None], 0, 1))
    polys, cols, depths = np.concatenate(polys), np.concatenate(cols), np.concatenate(depths)
    order = np.argsort(-depths)
    ax.add_collection(PolyCollection(polys[order], facecolors=cols[order], edgecolors="none", antialiased=False))
    allxy = polys.reshape(-1, 2)
    ax.set_xlim(allxy[:, 0].min() - 5, allxy[:, 0].max() + 5); ax.set_ylim(allxy[:, 1].min() - 5, allxy[:, 1].max() + 5)
    ax.set_aspect("equal"); ax.axis("off")


def share_limits(axes):
    lo_x = min(a.get_xlim()[0] for a in axes); hi_x = max(a.get_xlim()[1] for a in axes)
    lo_y = min(a.get_ylim()[0] for a in axes); hi_y = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_xlim(lo_x, hi_x); a.set_ylim(lo_y, hi_y)


def section(P, F, x0=0.0):
    d = P[:, 0] - x0
    segs = []
    for a, b, c in F:
        pts = []
        for i, j in ((a, b), (b, c), (c, a)):
            if (d[i] < 0) != (d[j] < 0):
                q = P[i] + d[i] / (d[i] - d[j]) * (P[j] - P[i])
                pts.append((-q[2], q[1]))
        if len(pts) == 2:
            segs.append(pts)
    return segs


def render(subject, out=None):
    tpl = config.template_export_dir()
    sub = tpl / "subjects" / subject
    out = Path(out) if out else sub / "views"
    out.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(sub / "report.json"))["metrics"]
    s = rep["scale"]
    _V, F, _ = fx.load_skin()
    # The head the fit started from (the human base head, or the lab head), at subject scale.
    from pipeline import fit_subject as fs
    start_name = rep.get("start_head", "lab head")
    start_bodies = (fs.load_base_bodies(start_name) if start_name != "lab head" else fs.load_template_bodies())[0]
    V = start_bodies[fx.SKIN]["out"]
    fit = np.load(sub / "skin_fit_output.npz")
    X, cloud = fit["vertices"], fit["cloud"]
    written = []

    # 1. before / scan / after, front and side, true size
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), facecolor=BG)
    for row, view in enumerate(["front", "side"]):
        draw(axes[row, 0], V, F, view)
        xy, depth = project(cloud, view)
        order = np.argsort(-depth)
        axes[row, 1].scatter(xy[order, 0], xy[order, 1], s=0.6, c=-depth[order], cmap="Blues")
        axes[row, 1].set_aspect("equal"); axes[row, 1].axis("off")
        draw(axes[row, 2], X, F, view)
        share_limits(axes[row])
    for col, t in enumerate(["Start head ({0}), original size".format(start_name), "Phone scan (face only)",
                             "Lab skin fitted ({0:.1f}% size)".format(100 * s)]):
        axes[0, col].set_title(t, fontsize=13)
    plt.tight_layout(); plt.savefig(out / "fit_compare.png", dpi=80, facecolor=BG); plt.close()
    written.append("fit_compare.png")

    # 2. scan on skin, reshaping, folds
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor=BG)
    draw(axes[0], X, F, "side")
    xy, _ = project(cloud[::3], "side")
    axes[0].scatter(xy[:, 0], xy[:, 1], s=0.4, c="#1f5fa8", alpha=0.6)
    axes[0].set_title("Fitted skin (side), scan points in blue", fontsize=12)
    d = np.linalg.norm(X - s * V, axis=1)
    vm = float(np.percentile(d, 99))
    draw(axes[1], X, F, "front", values=d, vmax=vm)
    axes[1].set_title("Reshaping after the uniform scale (0 to {0:.0f} mm)".format(vm), fontsize=12)
    fl = geo.flipped_faces(s * V, X, F, min_area_ratio=0.05)
    draw(axes[2], X, F, "front")
    xy, _ = project(X[F[fl]].mean(axis=1), "front")
    axes[2].scatter(xy[:, 0], xy[:, 1], s=3, c="#c0392b")
    axes[2].set_title("Folded triangles in red ({0:,} of {1:,})".format(int(fl.sum()), len(F)), fontsize=12)
    plt.tight_layout(); plt.savefig(out / "fit_detail.png", dpi=80, facecolor=BG); plt.close()
    written.append("fit_detail.png")

    # 3. inside the head: midline cut, before dashed, after solid
    man = json.load(open(tpl / "manifest.json"))["bodies"]
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    for key, info in man.items():
        if key.startswith("jack01") or key == "skin_cloth" or not (sub / (key + ".npz")).exists():
            continue
        zb = np.load(tpl / (key + ".npz"))
        Fb = zb["faces"].reshape(-1, 3)
        before = s * (start_bodies[key]["out"] if key == fx.SKIN else start_bodies[key]["rest"])
        after = X if key == fx.SKIN else np.load(sub / (key + ".npz"))["rest_vertices"]
        lw = 2.2 if key == fx.SKIN else 1.4
        ax.add_collection(LineCollection(section(before, Fb), colors=COLOUR[info["role"]], linewidths=lw * 0.7,
                                         linestyles=(0, (3, 2)), alpha=0.55))
        ax.add_collection(LineCollection(section(after, Fb), colors=COLOUR[info["role"]], linewidths=lw))
    ax.set_aspect("equal"); ax.set_xlim(-170, 60); ax.set_ylim(-125, 120); ax.axis("off")
    ax.legend(handles=[Line2D([], [], color=COLOUR["cloth"], lw=2.2, label="skin"),
                       Line2D([], [], color=COLOUR["tissue"], lw=1.4, label="muscle and fat"),
                       Line2D([], [], color=COLOUR["bone"], lw=1.4, label="skull, jaw, teeth"),
                       Line2D([], [], color="#555", lw=1, ls=(0, (3, 2)), label="before fit (start head, scaled)"),
                       Line2D([], [], color="#555", lw=1.6, label="after fit")],
              loc="lower left", frameon=False, fontsize=11)
    plt.tight_layout(); plt.savefig(out / "inside.png", dpi=80, facecolor=BG); plt.close()
    written.append("inside.png")

    # 4. skin, eyeballs and teeth together, as Maya shows them: anything poking
    # through the skin (eyeballs, teeth) shows in front of it
    parts = [(X, F, SKIN_RGB)]
    for key, rgb in (("middleres_Eyes_l_simple", (0.93, 0.93, 0.9)), ("middleres_Eyes_r_simple", (0.93, 0.93, 0.9)),
                     ("middleres_TeethUp", (0.95, 0.92, 0.82)), ("middleres_TeethDown", (0.95, 0.92, 0.82))):
        if (sub / (key + ".npz")).exists():
            z = np.load(sub / (key + ".npz"))
            parts.append((z["rest_vertices"], z["faces"].reshape(-1, 3), np.array(rgb)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor=BG)
    for ax, view in zip(axes, ("front", "side")):
        draw_parts(ax, parts, view)
    axes[0].set_title("Skin, eyeballs and teeth together", fontsize=12)
    plt.tight_layout(); plt.savefig(out / "parts.png", dpi=80, facecolor=BG); plt.close()
    written.append("parts.png")

    # 5. animations (if simulated): the fitted skin, and how far it has moved
    for name, label in (("anim_fit", "Lab scene frames 0-50: the tissue settling (no speech)"),
                        ("anim_ui_fit", '"not even you" from the keyboard UI')):
        f = sub / (name + ".npz")
        if not f.exists():
            continue
        frames = np.load(f)["skin"]
        written += animate(frames, F, out / (name + ".gif"), out / (name + "_sheet.png"), label)
    return out, written


def animate(frames, F, gif, sheet, label, step=2):
    from PIL import Image
    move = np.linalg.norm(frames - frames[:1], axis=2)
    vmax = max(float(np.percentile(move, 99.5)), 0.5)
    imgs = []
    idx = list(range(0, len(frames), step))
    for t in idx:
        fig, axes = plt.subplots(1, 2, figsize=(9, 5.4), facecolor=BG)
        draw(axes[0], frames[t], F, "front")
        draw(axes[1], frames[t], F, "front", values=move[t], vmax=vmax, cmap="magma")
        share_limits(axes)
        axes[0].set_title("frame {0}".format(t), fontsize=11)
        axes[1].set_title("moved from frame 0 (0 to {0:.1f} mm)".format(vmax), fontsize=11)
        fig.suptitle(label, fontsize=12)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
        plt.close(fig)
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 * step / 30), loop=0)
    picks = [imgs[i] for i in np.linspace(0, len(imgs) - 1, 6).astype(int)]
    w, h = picks[0].size
    sheet_img = Image.new("RGB", (w * 3, h * 2), BG)
    for k, im in enumerate(picks):
        sheet_img.paste(im, ((k % 3) * w, (k // 3) * h))
    sheet_img.save(sheet)
    return [gif.name, sheet.name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out, written = render(args.subject, args.out)
    print("wrote", out, written)


if __name__ == "__main__":
    main()
