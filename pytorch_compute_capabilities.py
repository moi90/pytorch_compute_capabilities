import fnmatch
import glob
import json
import multiprocessing
import os
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from typing import List, Mapping

import pandas as pd
import parse
import tqdm

BASE_URL = "https://conda.anaconda.org/pytorch/linux-64/"


def strip_extension(fn: str, extensions=[".tar.bz2", ".tar.gz"]):
    for ext in extensions:
        if fn.endswith(ext):
            return fn[: -len(ext)]
    raise ValueError(f"Unexpected extension for filename: {fn}")


def download_file(pkg_archive_fn):
    if os.path.isfile(pkg_archive_fn):
        return

    file_url = urllib.parse.urljoin(BASE_URL, pkg_archive_fn)
    bar = tqdm.tqdm(desc=f"Downloading {pkg_archive_fn}...")

    def _update_bar(blocks_transferred, block_size, total_size):
        bar.total = total_size // 1024
        bar.update(block_size // 1024)

    urllib.request.urlretrieve(file_url, pkg_archive_fn, _update_bar)

    bar.close()


def get_lib_fns(pkg_archive_fn) -> List[str]:
    pkg_name = strip_extension(pkg_archive_fn)

    os.makedirs(pkg_name, exist_ok=True)

    lib_fns = glob.glob(os.path.join(pkg_name, "*.so"))

    if lib_fns:
        return lib_fns

    # Else download and extract
    download_file(pkg_archive_fn)

    try:
        print(f"Reading archive {pkg_archive_fn}...")
        with tarfile.open(pkg_archive_fn, "r:*") as tf:
            match = False
            for m in tf:
                libname = os.path.basename(m.name)
                if fnmatch.fnmatch(libname, "*.so"):
                    print(f"Extracting {libname}...")
                    with open(os.path.join(pkg_name, libname), "wb") as df:
                        shutil.copyfileobj(tf.extractfile(m), df)
                        match = True

            if not match:
                print("*.so not found")
                with open(os.path.join(pkg_name, "filelist.txt"), "w") as f:
                    f.write("\n".join(tf.getnames()))
    except (tarfile.TarError, EOFError) as exc:
        print(exc)
        os.remove(pkg_archive_fn)
        return []
    else:
        return glob.glob(os.path.join(pkg_name, "*.so"))


def get_summary(pkg_archive_fn) -> Mapping[str, str]:

    pkg_name = strip_extension(pkg_archive_fn)

    summary_fn = os.path.join(pkg_name, "summary.json")

    try:
        with open(summary_fn) as f:
            return json.load(f)
    except FileNotFoundError:
        pass

    architectures = set()

    lib_fns = get_lib_fns(pkg_archive_fn)

    for lib_fn in lib_fns:
        print(f"Reading lib {lib_fn}...")
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

    pkg_name = strip_extension(pkg_archive_fn)
    summary = {"package": pkg_name, "architectures": ", ".join(sorted(architectures))}

    # Cleanup package archive
    if architectures:
        with open(summary_fn, "w") as f:
            json.dump(summary, f)

        try:
            os.remove(pkg_archive_fn)
        except FileNotFoundError:
            pass

        for fn in lib_fns:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    return summary


def main():
    print("Reading repodata.json...")

    download_file("repodata.json")

    with open("repodata.json") as f:
        repodata = json.load(f)

    pkg_archive_fns = []
    for pkg_archive_fn, p in repodata["packages"].items():
        if p["name"] != "pytorch":
            continue

        if "cuda" not in p["build"]:
            continue

        if "py3.7" not in p["build"]:
            continue

        pkg_archive_fns.append(pkg_archive_fn)

    print("Processing", ", ".join(pkg_archive_fns), "...")
    print()

    with multiprocessing.Pool(4) as p:
        table = list(p.imap_unordered(get_summary, pkg_archive_fns))

    table = pd.DataFrame(table)
    table = table.sort_values("package")
    with open("table.md", "w") as f:
        table.to_markdown(f, tablefmt="github", index=False)


if __name__ == "__main__":
    main()
