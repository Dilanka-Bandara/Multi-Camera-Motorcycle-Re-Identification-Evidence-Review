"""Download public evaluation archives and extract them without replacing existing files."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def extract(archive: Path, destination: Path) -> None:
    """Reject escaping paths and resume only byte-identical completed files."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in tqdm(bundle.infolist(), desc="Extract"):
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root) or item.filename.startswith(("/", "\\")):
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with bundle.open(item) as source, target.open("rb") as existing:
                    if (
                        hashlib.file_digest(source, "sha256").digest()
                        != hashlib.file_digest(existing, "sha256").digest()
                    ):
                        raise FileExistsError(target)
                continue
            temporary = target.with_name(target.name + ".extracting")
            with bundle.open(item) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            temporary.rename(target)


def download_zenodo(root: Path) -> Path:
    path = root / "Vehicle_Tracking.zip"
    if not path.exists():
        part = path.with_suffix(".zip.part")
        start = part.stat().st_size if part.exists() else 0
        url = "https://zenodo.org/records/18195750/files/Vehicle_Tracking.zip?download=1"
        with requests.get(
            url,
            stream=True,
            headers={"Range": f"bytes={start}-"} if start else {},
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            resume = start > 0 and response.status_code == 206
            if resume and not response.headers.get("Content-Range", "").startswith(
                f"bytes {start}-"
            ):
                raise ValueError("Server returned an unexpected resume offset")
            initial = start if resume else 0
            total = int(response.headers.get("Content-Length", 0)) + initial
            with (
                part.open("ab" if resume else "wb") as stream,
                tqdm(total=total, initial=initial, unit="B", unit_scale=True) as bar,
            ):
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
                    bar.update(len(chunk))
        part.rename(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != "a1a511436e55b8168928a6a278ae2ba7":
        raise ValueError("Vehicle Tracking archive checksum differs from the published record")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["roundabouthd", "vntraffic"], required=True)
    parser.add_argument("--out", type=Path, default=Path("data/user/datasets"))
    args = parser.parse_args()
    root = args.out / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    if args.dataset == "roundabouthd":
        archive = Path(
            hf_hub_download(
                repo_id="yl4300/RoundaboutHD",
                repo_type="dataset",
                filename="RoundaboutHD.zip",
                local_dir=root,
                token=False,
            )
        )
    else:
        archive = download_zenodo(root)
    extract(archive, root / "extracted")
    print(
        json.dumps(
            {"dataset": args.dataset, "archive": str(archive), "bytes": archive.stat().st_size}
        )
    )


if __name__ == "__main__":
    main()
