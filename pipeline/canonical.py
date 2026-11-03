"""MediaPipe's canonical face mesh, read out of the face_landmarker.task bundle.

The bundle is a zip; `geometry_pipeline_metadata_landmarks.binarypb` inside it
is a GeometryPipelineMetadata protobuf:

    1  canonical_mesh            Mesh3d { 1 vertex_type, 2 primitive_type,
                                          3 vertex_buffer: float,
                                          4 index_buffer:  uint32 }
    2  procrustes_landmark_basis repeated { 1 landmark_id: uint32, 2 weight: float }
    3  input_source              enum

(Field numbers read off the bundle itself; the buffers are stored unpacked,
but packed encoding is accepted too.)

Vertex type VERTEX_PT stores 5 floats per vertex (x, y, z, u, v), in cm. Parsed
by hand so mayapy needs no protobuf package.

    mayapy pipeline/canonical.py     # writes config/mediapipe_canonical.npz
"""

import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402

N_CANONICAL = 468          # the canonical mesh has no iris points
N_LANDMARKS = 478          # 468 + 2 x 5 iris landmarks
IRIS = list(range(468, 478))
CM_TO_MM = 10.0

# Contours from mediapipe/python/solutions/face_mesh_connections.py, flattened
# to vertex sets. "Left"/"right" are MediaPipe's: the subject's own sides.
LIPS_OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_OUTER_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
RIGHT_EYE_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LEFT_EYE_UPPER = [263, 466, 388, 387, 386, 385, 384, 398, 362]
LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]

# Named anchors (verified against the canonical mesh by tests/test_pipeline.py:
# nose tip is the most forward point, forehead above chin, 33 and 263 mirror).
NOSE_TIP, NOSE_BRIDGE, NASION = 1, 4, 6
FOREHEAD, CHIN, CHIN_UPPER = 10, 152, 199
FACE_EDGE_R, FACE_EDGE_L = 234, 454
EYE_R_OUTER, EYE_R_INNER, EYE_L_INNER, EYE_L_OUTER = 33, 133, 362, 263
MOUTH_R, MOUTH_L = 61, 291


def _varint(buf, i):
    out, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _varint(buf, i)
        elif wt == 1:
            val, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(buf, i)
            val, i = buf[i:i + n], i + n
        elif wt == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            raise ValueError("unsupported wire type {0}".format(wt))
        yield num, wt, val


def _packed_varints(buf):
    out, i = [], 0
    while i < len(buf):
        v, i = _varint(buf, i)
        out.append(v)
    return out


def parse_metadata(blob):
    """-> dict(vertices (468,3) mm, uv (468,2), faces (F,3), basis {id: weight})."""
    floats, indices, basis = [], [], {}
    for num, wt, val in _fields(blob):
        if num == 2 and wt == 2:
            lid, w = None, None
            for n2, w2, v2 in _fields(val):
                if n2 == 1:
                    lid = v2
                elif n2 == 2:
                    w = struct.unpack("<f", v2)[0]
            basis[int(lid)] = float(w)
        elif num == 1 and wt == 2:
            for n2, w2, v2 in _fields(val):
                if n2 == 3 and w2 == 2:
                    floats.extend(struct.unpack("<{0}f".format(len(v2) // 4), v2))
                elif n2 == 3 and w2 == 5:
                    floats.append(struct.unpack("<f", v2)[0])
                elif n2 == 4 and w2 == 2:
                    indices.extend(_packed_varints(v2))
                elif n2 == 4 and w2 == 0:
                    indices.append(v2)
    vb = np.asarray(floats, dtype=np.float64)
    if len(vb) != N_CANONICAL * 5:
        raise ValueError("vertex buffer holds {0} floats, expected {1}".format(len(vb), N_CANONICAL * 5))
    if len(indices) % 3:
        raise ValueError("index buffer length {0} is not a multiple of 3".format(len(indices)))
    vb = vb.reshape(N_CANONICAL, 5)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if faces.max() >= N_CANONICAL:
        raise ValueError("face index out of range")
    return {"vertices": vb[:, :3] * CM_TO_MM, "uv": vb[:, 3:], "faces": faces, "basis": basis}


def extract(task_path=None):
    with zipfile.ZipFile(task_path or config.MEDIAPIPE_TASK) as z:
        blob = z.read("geometry_pipeline_metadata_landmarks.binarypb")
    return parse_metadata(blob)


def load(path=None):
    """The committed canonical mesh: vertices in mm, MediaPipe's own frame
    (Y-up, face toward +Z, subject's right at -X)."""
    with np.load(path or config.MEDIAPIPE_CANONICAL_NPZ, allow_pickle=False) as z:
        return {"vertices": z["vertices"], "uv": z["uv"], "faces": z["faces"],
                "basis": dict(zip(z["basis_ids"].tolist(), z["basis_weights"].tolist()))}


def main():
    m = extract()
    ids = np.array(sorted(m["basis"]), dtype=np.int64)
    np.savez(config.MEDIAPIPE_CANONICAL_NPZ, vertices=m["vertices"], uv=m["uv"], faces=m["faces"],
             basis_ids=ids, basis_weights=np.array([m["basis"][i] for i in ids]),
             source="face_landmarker.task:geometry_pipeline_metadata_landmarks.binarypb",
             units="mm", licence="Apache-2.0 (MediaPipe)")
    v = m["vertices"]
    print("canonical: {0} verts, {1} faces, {2} Procrustes basis landmarks".format(
        len(v), len(m["faces"]), len(ids)))
    print("extent mm: {0}".format(np.round(v.max(0) - v.min(0), 1).tolist()))
    print("nose tip (1) {0}, forehead (10) {1}, chin (152) {2}, 33 {3}, 263 {4}".format(
        *[np.round(v[i], 1).tolist() for i in (1, 10, 152, 33, 263)]))
    print("wrote", config.MEDIAPIPE_CANONICAL_NPZ)


if __name__ == "__main__":
    main()
