#!/usr/bin/env python3
"""Draw Devanagari and Arabic strings in matplotlib with correct shaping.

Matplotlib lays text out one codepoint at a time: no conjunct formation, no
matra reordering, no cursive joining.  Devanagari and Arabic therefore come out
as a row of disconnected letters that a native reader cannot read.  This module
shapes the string with HarfBuzz and emits the resulting glyph outlines as a
single matplotlib ``Path``, so the label is drawn as vector art rather than as
text.  The cost is that these labels are not selectable in the PDF; the benefit
is that they are correct.

    p = ShapedFont(FONT).text_path("स्त्री", 8)   # Path, in points, baseline at y=0
"""
from __future__ import annotations

import functools

import numpy as np
import uharfbuzz as hb
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

MOVETO, LINETO, CURVE3, CURVE4, CLOSE = (
    Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE4, Path.CLOSEPOLY)


class _PathPen:
    """A fontTools pen that accumulates matplotlib vertices/codes."""

    def __init__(self):
        self.verts, self.codes = [], []
        self._start = None

    def moveTo(self, pt):
        self.verts.append(pt), self.codes.append(MOVETO)
        self._start = pt

    def lineTo(self, pt):
        self.verts.append(pt), self.codes.append(LINETO)

    def curveTo(self, *pts):
        for p in pts:                      # cubic: always 2 controls + endpoint
            self.verts.append(p), self.codes.append(CURVE4)

    def qCurveTo(self, *pts):
        if pts[-1] is None:                # all-off-curve contour: start at a midpoint
            pts = pts[:-1]
            start = tuple((np.array(pts[0]) + np.array(pts[-1])) / 2)
            self.moveTo(start)
            pts = pts + (start,)
        off = list(pts[:-1])
        for a, b in zip(off, off[1:]):     # implied on-curve point between two controls
            mid = tuple((np.array(a) + np.array(b)) / 2)
            self.verts += [a, mid]
            self.codes += [CURVE3, CURVE3]
        if off:
            self.verts += [off[-1], pts[-1]]
            self.codes += [CURVE3, CURVE3]
        else:
            self.lineTo(pts[-1])

    def closePath(self):
        if self.verts:
            self.verts.append(self._start or self.verts[-1]), self.codes.append(CLOSE)

    endPath = closePath

    def addComponent(self, name, transform):  # composites are flattened upstream
        pass


class ShapedFont:
    """HarfBuzz shaping + glyph outlines for one font file."""

    def __init__(self, path: str):
        self.path = path
        blob = hb.Blob.from_file_path(path)
        self.hb_font = hb.Font(hb.Face(blob))
        self.tt = TTFont(path, fontNumber=0, lazy=True)
        self.upem = self.tt["head"].unitsPerEm
        self._glyphs = self.tt.getGlyphSet()
        self._order = self.tt.getGlyphOrder()

    @functools.lru_cache(maxsize=2048)
    def _glyph_path(self, gid: int):
        pen = _PathPen()
        try:  # DecomposingRecordingPen flattens composite glyphs for us
            rec = DecomposingRecordingPen(self._glyphs)
            self._glyphs[self._order[gid]].draw(rec)
            rec.replay(pen)
        except Exception:
            return None
        if not pen.verts:
            return None
        return Path(np.asarray(pen.verts, float), np.asarray(pen.codes, np.uint8))

    def text_path(self, text: str, size: float, direction: str | None = None,
                  script: str | None = None, features: dict | None = None,
                  pad_metrics: bool = True):
        """Return (Path in points with baseline at y=0, advance width in points).

        With ``pad_metrics`` the path carries an invisible zero-width strut from
        the descender to the ascender, so that every string has the same height
        and rows of labels line up instead of jittering with their descenders.
        """
        buf = hb.Buffer()
        buf.add_str(text)
        if direction:
            buf.direction = direction
        if script:
            buf.script = script
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, features)

        s = size / self.upem
        verts, codes, pen_x, pen_y = [], [], 0.0, 0.0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            gp = self._glyph_path(info.codepoint)
            if gp is not None and len(gp.vertices):
                t = Affine2D().translate(pen_x + pos.x_offset, pen_y + pos.y_offset).scale(s)
                tp = t.transform_path(gp)
                verts.append(tp.vertices)
                codes.append(tp.codes)
            pen_x += pos.x_advance
            pen_y += pos.y_advance
        if pad_metrics:
            hhea = self.tt["hhea"]
            # Two bare MOVETOs: they extend the path's bbox but describe no
            # segment, so nothing is drawn (a LINETO here shows up as a hairline
            # in some PDF rasterisers).
            verts.append(np.array([[0.0, hhea.descent * s], [0.0, hhea.ascent * s]]))
            codes.append(np.array([MOVETO, MOVETO], np.uint8))
        if not verts:
            return Path(np.zeros((0, 2))), pen_x * s
        return Path(np.concatenate(verts), np.concatenate(codes)), pen_x * s


def native_label(ax, text, font: ShapedFont, size, x, y, *, xycoords, color="#333333",
                 ha="right", va="center", dx=0.0, dy=0.0, script=None, direction=None):
    """Place a shaped string at (x, y); dx/dy are an offset in points."""
    path, width = font.text_path(text, size, direction=direction, script=script)
    patch = PathPatch(path, facecolor=color, edgecolor="none", lw=0)
    # points -> inches -> device, via the figure's live dpi transform, so the
    # label keeps its size when the figure is saved at a different dpi.
    box = AuxTransformBox(Affine2D().scale(1 / 72.0) + ax.figure.dpi_scale_trans)
    box.add_artist(patch)
    # AuxTransformBox sizes itself from the artists' extents; anchor on the text
    # box so that `ha`/`va` behave like they do for a normal Text.
    ab = AnnotationBbox(
        box, (x, y), xycoords=xycoords, xybox=(dx, dy), boxcoords="offset points",
        box_alignment=({"left": 0.0, "center": 0.5, "right": 1.0}[ha],
                       {"bottom": 0.0, "center": 0.5, "top": 1.0}[va]),
        frameon=False, pad=0.0, annotation_clip=False,
    )
    ab.set_zorder(5)
    ax.add_artist(ab)
    return width
