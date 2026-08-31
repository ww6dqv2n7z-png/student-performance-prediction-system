"""Download the official UCI Student Performance archive with safe extraction."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
ALLOWED = {"student-mat.csv", "student-por.csv", "student.txt"}
MAX_DOWNLOAD = 2 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/uci")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "student-performance.zip"
    request = urllib.request.Request(URL, headers={"User-Agent": "MTU-CEIT-Research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise ValueError("UCI archive exceeded the expected safety limit")
    archive.write_bytes(data)
    with zipfile.ZipFile(archive) as outer:
        inner_name = next((name for name in outer.namelist() if Path(name).name == "student.zip"), None)
        if inner_name is None:
            raise ValueError("Official UCI archive did not contain student.zip")
        inner_data = outer.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(inner_data)) as zipped:
        available = {Path(name).name: name for name in zipped.namelist()}
        for filename in ALLOWED:
            if filename in available:
                target = output / filename
                target.write_bytes(zipped.read(available[filename]))
    print(f"Downloaded official UCI files to {output}")


if __name__ == "__main__":
    main()
