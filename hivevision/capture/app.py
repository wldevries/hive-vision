"""FastAPI app for the capture/label workflow.

Serves an inbox browser and a per-photo label page. The user drops phone photos
in ``data/store/inbox/`` and marks each tile's icon center with its class; on
save the photo is EXIF-normalized and a row is appended to ``data/labels.jsonl``
(see ``hivevision.data.store``). Single local user, so there is no auth and the
store is the durable output.

The label page does direct manual marking today; the 4-click homography
auto-projection (``hivevision.geometry``) is a planned fast-follow that drops
onto the same page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hivevision.data.store import LabelStore
from hivevision.geometry import axial_centers, project, recover_lattice
from hivevision.pieces import CLASSES, class_name, is_valid_class

STATIC_DIR = Path(__file__).parent / "static"


class PointIn(BaseModel):
    label: str
    x: float
    y: float


class LabelIn(BaseModel):
    """Save payload: the inbox ``src`` and its marked points (normalized frame)."""

    src: str
    points: list[PointIn]


class RecoverIn(BaseModel):
    """Recover the board from the current points (for the live board preview)."""

    points: list[PointIn]


def create_app(root: Path) -> FastAPI:
    store = LabelStore(root=root)
    app = FastAPI(title="HiveVision capture")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/label")
    def label_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "label.html")

    @app.get("/api/classes")
    def classes() -> JSONResponse:
        return JSONResponse([{"code": c, "name": class_name(c)} for c in CLASSES])

    @app.get("/api/inbox")
    def inbox() -> JSONResponse:
        return JSONResponse(store.list_inbox())

    @app.get("/api/image")
    def image(src: str) -> Response:
        try:
            return Response(store.normalized_bytes(src), media_type="image/jpeg")
        except FileNotFoundError as e:
            raise HTTPException(404, f"no such inbox photo: {src}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/thumb")
    def thumb(src: str, w: int = 320) -> Response:
        """Cached downscaled JPEG for the inbox grid (the full photo is megabytes)."""
        w = max(64, min(w, 1024))
        try:
            return Response(store.thumb_bytes(src, max_w=w), media_type="image/jpeg")
        except FileNotFoundError as e:
            raise HTTPException(404, f"no such inbox photo: {src}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/label")
    def get_label(src: str) -> JSONResponse:
        return JSONResponse(store.get_label(src))

    @app.post("/api/label")
    def save_label(payload: LabelIn) -> JSONResponse:
        bad = [p.label for p in payload.points if not is_valid_class(p.label)]
        if bad:
            raise HTTPException(400, f"unknown class codes: {sorted(set(bad))}")
        try:
            row = store.save_label(
                payload.src, [{"label": p.label, "x": p.x, "y": p.y} for p in payload.points]
            )
        except FileNotFoundError as e:
            raise HTTPException(404, f"no such inbox photo: {payload.src}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return JSONResponse(row)

    @app.post("/api/recover")
    def recover(payload: RecoverIn) -> JSONResponse:
        """Recover the board (q, r) from the marked centers, for the live preview."""
        if len(payload.points) < 3:
            return JSONResponse({"ok": False, "reason": "need at least 3 tiles"})
        pts = np.array([[p.x, p.y] for p in payload.points], dtype=np.float64)
        try:
            fit = recover_lattice(pts)
        except ValueError as e:
            return JSONResponse({"ok": False, "reason": str(e)})

        # Orient the preview like the photo: take the homography's local rotation
        # at the board centre (its Jacobian, nearest-orthogonal part) and lay the
        # regular board out with that rotation — same orientation as the photo,
        # minus the perspective/shear.
        canon = axial_centers([(int(c[0]), int(c[1])) for c in fit.axial], size=1.0)
        c0 = canon.mean(axis=0)
        o = project(fit.homography, c0[None])[0]
        jac = np.array(
            [
                project(fit.homography, (c0 + [1, 0])[None])[0] - o,
                project(fit.homography, (c0 + [0, 1])[None])[0] - o,
            ]
        ).T  # columns = image displacement per unit plane x / y
        u, _, vt = np.linalg.svd(jac)
        rot = u @ vt  # nearest orthogonal (orientation incl. any flip)
        disp = (canon - c0) @ rot.T
        orient_deg = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

        placements = [
            {
                "label": payload.points[i].label,
                "q": int(c[0]),
                "r": int(c[1]),
                "dx": float(disp[i, 0]),
                "dy": float(disp[i, 1]),
            }
            for i, c in enumerate(fit.axial)
        ]
        return JSONResponse(
            {
                "ok": True,
                "placements": placements,
                "residual_frac": fit.residual_frac,
                "n": fit.n,
                "orient_deg": orient_deg,
            }
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
