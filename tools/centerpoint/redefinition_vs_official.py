"""Gate: the plain-PyTorch CenterPoint re-definition vs the official mmdetection3d head outputs (tools only).

For the first N dumped samples of ``official_run.py`` (official voxelizer output and raw CenterHead outputs), runs
``centerpoint_model`` on (a) the official pillars as they are and (b) the same pillars from the SDV-BMT payload padded
to 40,000, and reports per head tensor max|a-b| / max|b|.  Criterion (plan v02): <= 1e-4.

    <venv-mmseg>/bin/python tools/centerpoint/redefinition_vs_official.py --dump DIR --index INDEX --checkpoint CKPT --out OUT.json
"""
import argparse, hashlib, json, struct, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
import centerpoint_model as M

H = struct.Struct("<8s8I")


def payload_tensors(path: Path):
    b = path.read_bytes()
    _, P, MP, MPTS, F, *_ = H.unpack_from(b)
    o = H.size
    v = np.frombuffer(b, "<f4", MP * MPTS * F, o).reshape(MP, MPTS, F); o += v.nbytes
    n = np.frombuffer(b, "<i4", MP, o); o += n.nbytes
    yx = np.frombuffer(b, "<i4", MP * 2, o).reshape(MP, 2)
    return torch.from_numpy(v.copy()), torch.from_numpy(n.copy()), torch.from_numpy(yx.copy()), P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--index", required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = M.load(a.checkpoint)
    rows = [json.loads(l) for l in open(a.index)]
    idx_dir = Path(a.index).parent
    worst, per = 0.0, []
    for f in sorted(Path(a.dump).glob("*.npz")):
        i = int(f.stem)
        z = np.load(f)
        vox, nump, coors = (torch.from_numpy(z[k]) for k in ("voxels", "num_points", "coors"))
        with torch.no_grad():
            ours = model.heads(vox, nump, coors[:, 2:4].contiguous())
            pv, pn, pyx, P = payload_tensors(idx_dir / rows[i]["payload_file"])
            padded = model(pv, pn, pyx)
        unpadded_cat = torch.cat([d[n] for d in ours for n in [h for h, _ in M.COMMON_HEADS] + ["heatmap"]], 1)
        rec = {"index": i, "pillars": int(P), "padded_equals_unpadded": bool(torch.equal(padded, unpadded_cat))}
        for t, d in enumerate(ours):
            for n, val in d.items():
                ref = torch.from_numpy(z[f"head:[{t}][0].{n}"])
                r = float((val - ref).abs().max() / ref.abs().max())
                rec[f"{t}.{n}"] = r
                worst = max(worst, r)
        per.append(rec)
        print(i, "max rel", max(v for k, v in rec.items() if "." in k), "padded==unpadded", rec["padded_equals_unpadded"])
    out = {"criterion": "max|ours-official|/max|official| per head tensor <= 1e-4", "worst": worst,
           "pass": worst <= 1e-4, "samples": per,
           "model_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("worst", worst, "pass", out["pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
