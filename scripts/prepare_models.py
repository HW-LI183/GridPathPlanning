#!/usr/bin/env python3
"""通过工作站镜像固定模型版本，下载并校验 PyTorch 所需文件。"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/models.json"
FILES = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "spiece.model", "model.safetensors",
)
HTTP_HEADERS = {"User-Agent": "curl/8.5.0"}


def read_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HTTP_HEADERS), timeout=30) as response:
        return json.load(response)


def save(manifest):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(MANIFEST)


def pin_versions(endpoint):
    project = json.loads((ROOT / "configs/project.json").read_text())
    manifest = {"endpoint": endpoint, "recorded_at": datetime.now(timezone.utc).isoformat(),
                "decision": "用户已确认使用原始 T5-small/base，不使用 Flan-T5", "models": []}
    for repo_id in project["models"]:
        metadata = read_json(f"{endpoint}/api/models/{repo_id}?blobs=true")
        revision = metadata["sha"]
        files = {item["rfilename"]: item for item in metadata["siblings"]}
        if "model.safetensors" not in files:
            raise RuntimeError(f"{repo_id} 缺少计划采用的 safetensors 权重")
        artifacts = []
        for name in FILES:
            if name not in files:
                continue
            item = files[name]
            lfs = item.get("lfs", {})
            artifacts.append({"name": name, "size": item.get("size", lfs.get("size")),
                              "expected_sha256": lfs.get("sha256"),
                              "git_blob_id": item.get("blobId"), "downloaded": False})
        entry = {"repo_id": repo_id, "revision": revision,
                 "local_directory": f"models/{repo_id.rsplit('/', 1)[-1]}", "files": artifacts}
        manifest["models"].append(entry)
        print(f"固定 {repo_id} @ {revision}，共 {len(artifacts)} 个文件", flush=True)
    save(manifest)
    return manifest


def verify_file(path, artifact):
    size = path.stat().st_size
    if artifact.get("size") is not None and size != artifact["size"]:
        raise ValueError(f"{path.name} 大小不符：{size} != {artifact['size']}")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_sha1.update(chunk)
    digest = sha256.hexdigest()
    expected = artifact.get("expected_sha256")
    if expected and digest != expected:
        raise ValueError(f"{path.name} SHA-256 校验失败")
    if not expected and artifact.get("git_blob_id") and git_sha1.hexdigest() != artifact["git_blob_id"]:
        raise ValueError(f"{path.name} Git blob 校验失败")
    return digest


def download(endpoint, entry, artifact):
    target = ROOT / entry["local_directory"] / artifact["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        artifact["sha256"] = verify_file(target, artifact)
        artifact["downloaded"] = True
        return
    partial = target.with_name(target.name + ".part")
    size = artifact.get("size")
    if partial.exists() and size is not None and partial.stat().st_size == size:
        artifact["sha256"] = verify_file(partial, artifact)
        partial.replace(target)
        artifact["downloaded"] = True
        return
    offset = partial.stat().st_size if partial.exists() else 0
    url = f"{endpoint}/{entry['repo_id']}/resolve/{entry['revision']}/{artifact['name']}"
    headers = dict(HTTP_HEADERS)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=30) as response:
        if offset and response.status == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("断点响应的 Content-Range 不匹配")
            mode = "ab"
        else:
            mode = "wb"
        artifact["resolved_host"] = urllib.parse.urlsplit(response.url).hostname
        with partial.open(mode) as stream:
            progress_mark = stream.tell() // (64 * 1024 * 1024)
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                if stream.tell() // (64 * 1024 * 1024) > progress_mark:
                    progress_mark = stream.tell() // (64 * 1024 * 1024)
                    print(f"下载 {entry['repo_id']}/{artifact['name']}: {stream.tell()} / {size} bytes", flush=True)
    artifact["sha256"] = verify_file(partial, artifact)
    partial.replace(target)
    artifact["downloaded"] = True
    artifact["download_seconds"] = round(time.monotonic() - started, 3)
    print(f"完成 {entry['repo_id']}/{artifact['name']} ({target.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="仅固定版本并下载配置和 tokenizer")
    args = parser.parse_args()
    endpoint = json.loads((ROOT / "configs/project.json").read_text())["hf_endpoint"].rstrip("/")
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else pin_versions(endpoint)
    if manifest["endpoint"].rstrip("/") != endpoint:
        raise ValueError("已固定清单的镜像与配置不一致，请显式审阅清单后再修改")
    expected_models = json.loads((ROOT / "configs/project.json").read_text())["models"]
    if [entry["repo_id"] for entry in manifest["models"]] != expected_models:
        raise ValueError("已固定模型清单与项目决策不一致")
    for entry in manifest["models"]:
        for artifact in entry["files"]:
            if args.metadata_only and artifact["name"] == "model.safetensors":
                continue
            download(endpoint, entry, artifact)
            save(manifest)
    print("模型清单：manifests/models.json；模型 revision 已固定，不会自动更新。")


if __name__ == "__main__":
    main()
