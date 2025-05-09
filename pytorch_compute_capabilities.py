# /// script
# dependencies = [
#   "natsort",
#   "packaging",
#   "parse",
#   "tqdm",
#   "pandas",
#   "tabulate",
# ]
# ///
import fnmatch
import glob
import json
import multiprocessing.pool
import os
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from typing import List, Mapping, Optional
from natsort import natsort_keygen
import packaging.version

import pandas as pd
import parse
import tqdm

BASE_URL = "https://conda.anaconda.org/pytorch/linux-64/"


def strip_extension(fn: str, extensions=[".tar.bz2", ".tar.gz"]):
    for ext in extensions:
        if fn.endswith(ext):
            return fn[: -len(ext)]
    raise ValueError(f"Unexpected extension for filename: {fn}")


def download_file(src, dst, force=False):
    if not force and os.path.isfile(dst):
        return dst

    src_url = urllib.parse.urljoin(BASE_URL, src)
    bar = tqdm.tqdm(
        desc=f"Downloading {src}...",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )

    def _update_bar(blocks_transferred, block_size, total_size):
        bar.total = total_size
        bar.update(block_size)

    urllib.request.urlretrieve(src_url, dst, _update_bar)

    bar.close()

    return dst


def get_lib_fns(raw_pkg_archive_fn) -> List[str]:
    pkg_name = strip_extension(raw_pkg_archive_fn)

    pkg_cache_dir = os.path.join("cache", pkg_name)
    os.makedirs(pkg_cache_dir, exist_ok=True)

    lib_fns = glob.glob(os.path.join(pkg_cache_dir, "*.so"))

    if lib_fns:
        return lib_fns

    # Else download and extract
    cache_pkg_archive_fn = os.path.join("cache", raw_pkg_archive_fn)

    try:
        download_file(raw_pkg_archive_fn, cache_pkg_archive_fn)
    except Exception as exc:
        tqdm.tqdm.write(str(exc))
        os.remove(cache_pkg_archive_fn)
        raise

    try:
        tqdm.tqdm.write(f"Reading archive {cache_pkg_archive_fn}...")
        with tarfile.open(cache_pkg_archive_fn, "r:*") as tf:
            match = False
            for m in tf:
                libname = os.path.basename(m.name)
                if fnmatch.fnmatch(libname, "*.so"):
                    tqdm.tqdm.write(f"Extracting {cache_pkg_archive_fn}/{libname}...")
                    with open(os.path.join(pkg_cache_dir, libname), "wb") as df:
                        shutil.copyfileobj(tf.extractfile(m), df)
                        match = True

            if not match:
                tqdm.tqdm.write(f"{cache_pkg_archive_fn}/*.so not found")
                with open(os.path.join(pkg_cache_dir, "filelist.txt"), "w") as f:
                    f.write("\n".join(tf.getnames()))
    except (tarfile.TarError, EOFError) as exc:
        tqdm.tqdm.write(str(exc))
        os.remove(cache_pkg_archive_fn)
        return []
    else:
        return glob.glob(os.path.join(pkg_cache_dir, "*.so"))


def get_cached_summary(pkg_archive_fn: str) -> Optional[Mapping[str, str]]:
    pkg_name = strip_extension(pkg_archive_fn)

    cache_dir = os.path.join("cache", pkg_name)
    summary_fn = os.path.join(cache_dir, "summary.json")

    try:
        with open(summary_fn) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_summary(pkg_archive_fn) -> Mapping[str, str]:
    summary = get_cached_summary(pkg_archive_fn)
    if summary is not None:
        return summary

    pkg_name = strip_extension(pkg_archive_fn)
    architectures = set()

    lib_fns = get_lib_fns(pkg_archive_fn)

    for lib_fn in lib_fns:
        tqdm.tqdm.write(f"Reading lib {lib_fn}...")
        try:
            output = subprocess.check_output(
                f'cuobjdump "{lib_fn}"', shell=True
            ).decode("utf-8")

            lib_archs = set(m["arch"] for m in parse.findall("arch = {arch}\n", output))

            if not lib_archs:
                os.remove(lib_fn)

            architectures.update(lib_archs)
        except subprocess.CalledProcessError:
            os.remove(lib_fn)

    cache_dir = os.path.join("cache", pkg_name)
    summary_fn = os.path.join(cache_dir, "summary.json")
    summary = {"package": pkg_name, "architectures": ", ".join(sorted(architectures))}

    # Cleanup package archive
    if architectures:
        with open(summary_fn, "w") as f:
            json.dump(summary, f)

        try:
            os.remove(os.path.join("cache", pkg_archive_fn))
        except FileNotFoundError:
            pass

        for fn in lib_fns:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    return summary


def parse_version(text):
    return packaging.version.parse(text)


parse_version.pattern = r"\d+(\.\d+)*"  # type: ignore

# See https://devguide.python.org/versions/ for supported Python versions
PYTHON_MIN_VER = packaging.version.parse("3.9")
PYTHON_MAX_VER = packaging.version.parse("4")


def main():
    # First of all, check that cuobjdump is available
    try:
        subprocess.check_output(
            "cuobjdump --version", shell=True, stderr=subprocess.STDOUT
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:
        print(exc.cmd)
        print(exc.output.decode("utf-8"))
        print(exc)
        print("cuobjdump not found. Please install the CUDA toolkit.")
        return

    print("Loading ")
    cached_repodata_fn = download_file(
        "repodata.json", os.path.join("cache", "repodata.json"), force=True
    )

    with open(cached_repodata_fn) as f:
        repodata = json.load(f)

    pkg_archive_fns = []
    for pkg_archive_fn, p in repodata["packages"].items():
        if p["name"] != "pytorch":
            continue

        python_ver = parse.search(
            "py{python_ver:ver}", p["build"], extra_types=dict(ver=parse_version)
        )
        if python_ver is None:
            continue

        if (PYTHON_MAX_VER < python_ver["python_ver"]) or (
            python_ver["python_ver"] < PYTHON_MIN_VER
        ):
            continue

        print(pkg_archive_fn, python_ver["python_ver"])

        if "cuda" not in p["build"]:
            continue

        if "py" not in p["build"]:
            continue

        pkg_archive_fns.append(pkg_archive_fn)

    print("Processing packages...")
    print()

    # Load cached summaries
    table = []
    remaining_pkg_archive_fns = []
    for pkg_archive_fn in pkg_archive_fns:
        summary = get_cached_summary(pkg_archive_fn)
        if summary is not None:
            table.append(summary)
        else:
            remaining_pkg_archive_fns.append(pkg_archive_fn)

    with multiprocessing.pool.ThreadPool(4) as p:
        table.extend(
            tqdm.tqdm(
                p.imap_unordered(get_summary, remaining_pkg_archive_fns),
                desc="Processing packages...",
                total=len(remaining_pkg_archive_fns),
            )
        )

    table = pd.DataFrame(table)
    table = table.sort_values("package", key=natsort_keygen(), ascending=False)

    with open("table.md", "w") as f:
        table.to_markdown(f, tablefmt="github", index=False)

    table.to_csv("table.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
