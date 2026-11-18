"""Official nuScenes info files for the CenterPoint cross-check (tools only; run with the cross-check env <venv-mmseg>).

Runs the official mmdetection3d 1.4.0 converter (``tools/dataset_converters/nuscenes_converter.create_nuscenes_infos``
with 10 sweeps, then ``update_infos_to_v2``) on a nuScenes data root and writes the v2 info files to OUT.  The
ground-truth database step of ``create_data.py`` (training only) is skipped.  Nothing here is used by the benchmark core.

``--only-present-scenes``: keep only the scenes whose LIDAR_TOP files are on disk (the converter's own availability
check, ``get_available_scenes``, tests the path type, not the file), and only their samples in ``nusc.sample`` (same
order; the sample token index is rebuilt), because ``_fill_trainval_infos`` checks the LiDAR file of every sample before
it looks at the split.  Used for v1.0-trainval when only the val LiDAR files were extracted: the train scenes drop out
and the val infos are produced by the unchanged converter code.

    <venv-mmseg>/bin/python tools/centerpoint/official_infos.py --root DATA_ROOT --version v1.0-mini --out OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import mmdet3d


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="nuscenes")
    ap.add_argument("--only-present-scenes", action="store_true")
    a = ap.parse_args()
    tools = Path(mmdet3d.__file__).parent / ".mim" / "tools"
    sys.path.insert(0, str(tools))  # the converters import each other as tools.dataset_converters.*
    sys.path.insert(0, str(tools.parent))
    from tools.dataset_converters import nuscenes_converter  # noqa: E402
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos  # noqa: E402

    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    if a.only_present_scenes:
        original = nuscenes_converter.get_available_scenes

        def present(nusc):
            keep = []
            for scene in original(nusc):
                sample = nusc.get("sample", scene["first_sample_token"])
                sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                if (root / sd["filename"]).is_file():
                    keep.append(scene)
            tokens = {scene["token"] for scene in keep}
            nusc.sample = [s for s in nusc.sample if s["scene_token"] in tokens]
            nusc._token2ind["sample"] = {s["token"]: i for i, s in enumerate(nusc.sample)}
            print(f"scenes with LiDAR files present: {len(keep)}, their samples: {len(nusc.sample)}", flush=True)
            return keep

        nuscenes_converter.get_available_scenes = present
    out.mkdir(parents=True, exist_ok=True)
    if not (out / f"{a.prefix}_infos_val_v1.pkl").exists():
        nuscenes_converter.create_nuscenes_infos(str(root), a.prefix, version=a.version, max_sweeps=10)
        for split in ("train", "val"):
            shutil.move(str(root / f"{a.prefix}_infos_{split}.pkl"), out / f"{a.prefix}_infos_{split}_v1.pkl")
    # update_infos_to_v2 opens NuScenes(dataroot='./data/nuscenes'): give it that path in a work directory
    work = out / "_work"
    (work / "data").mkdir(parents=True, exist_ok=True)
    link = work / "data" / "nuscenes"
    if not link.exists():
        os.symlink(root, link)
    os.chdir(work)
    import pickle
    for split in ("train", "val"):
        v2 = out / f"{a.prefix}_infos_{split}.pkl"
        with open(out / f"{a.prefix}_infos_{split}_v1.pkl", "rb") as fh:
            if not pickle.load(fh)["infos"]:  # update_infos_to_v2 fails on an empty split (unbound name)
                print(f"{split}: no infos, v2 update skipped", flush=True)
                continue
        shutil.copy(out / f"{a.prefix}_infos_{split}_v1.pkl", v2)
        update_pkl_infos("nuscenes", out_dir=str(out), pkl_path=str(v2))
    print("mmdet3d", mmdet3d.__version__, "infos in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
