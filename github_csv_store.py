"""Small GitHub Contents API helper for Streamlit CSV persistence.

Credentials are read from Streamlit secrets at runtime and are never written to
source files. This module performs network writes only when upload_file_bytes is
explicitly called by a user action in the app.
"""
from __future__ import annotations

import base64
import os
from typing import Any
from urllib.parse import quote

import requests

API_ROOT = "https://api.github.com"


def get_github_config(streamlit_secrets: Any = None) -> dict[str, str]:
    """Read repo settings from [github] Streamlit secrets or environment vars."""
    cfg: Any = {}
    if streamlit_secrets is not None:
        try:
            cfg = streamlit_secrets.get("github", {})
        except Exception:
            cfg = {}
    if not isinstance(cfg, dict):
        try:
            cfg = dict(cfg)
        except Exception:
            cfg = {}
    token = str(cfg.get("token") or os.getenv("GITHUB_TOKEN", "")).strip()
    owner = str(cfg.get("owner") or "tomlo6c316-git").strip()
    repo = str(cfg.get("repo") or "Horse_manus").strip()
    branch = str(cfg.get("branch") or "hkjc-data").strip()
    return {"token": token, "owner": owner, "repo": repo, "branch": branch}


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Horse-Manus-Streamlit-CSV-Store",
    }


def _contents_url(owner: str, repo: str, path: str = "") -> str:
    suffix = "/" + quote(path.strip("/"), safe="/") if path.strip("/") else ""
    return f"{API_ROOT}/repos/{quote(owner)}/{quote(repo)}/contents{suffix}"


def list_csv_files(owner: str, repo: str, folder: str, branch: str, token: str) -> list[dict[str, str]]:
    """List CSV files directly inside a repository folder (non-recursive)."""
    if not token:
        raise ValueError("未設定 GitHub token。")
    response = requests.get(
        _contents_url(owner, repo, folder),
        headers=_headers(token), params={"ref": branch}, timeout=25,
    )
    if response.status_code == 404:
        return []
    if not response.ok:
        raise RuntimeError(f"GitHub列出檔案失敗：HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    if not isinstance(payload, list):
        return []
    return [
        {"name": str(item.get("name", "")), "path": str(item.get("path", "")), "sha": str(item.get("sha", ""))}
        for item in payload
        if item.get("type") == "file" and str(item.get("name", "")).lower().endswith(".csv")
    ]


def download_file_bytes(owner: str, repo: str, path: str, branch: str, token: str) -> bytes:
    """Download a repository file through Contents API, including large CSVs."""
    if not token:
        raise ValueError("未設定 GitHub token。")
    response = requests.get(
        _contents_url(owner, repo, path),
        headers=_headers(token), params={"ref": branch}, timeout=40,
    )
    if not response.ok:
        raise RuntimeError(f"GitHub讀取檔案失敗：HTTP {response.status_code} {response.text[:300]}")
    payload = response.json()
    content = payload.get("content")
    if content:
        return base64.b64decode(content)
    download_url = payload.get("download_url")
    if not download_url:
        raise RuntimeError("GitHub沒有提供可下載的檔案內容。")
    raw = requests.get(download_url, headers={"Authorization": f"Bearer {token}", "User-Agent": "Horse-Manus-Streamlit-CSV-Store"}, timeout=60)
    if not raw.ok:
        raise RuntimeError(f"GitHub下載CSV失敗：HTTP {raw.status_code}")
    return raw.content


def upload_file_bytes(
    owner: str, repo: str, path: str, branch: str, token: str,
    contents: bytes, commit_message: str,
) -> dict[str, str]:
    """Create or replace a repository file; replacement includes the current SHA."""
    if not token:
        raise ValueError("未設定 GitHub token。")
    url = _contents_url(owner, repo, path)
    headers = _headers(token)
    current = requests.get(url, headers=headers, params={"ref": branch}, timeout=25)
    sha = None
    if current.status_code == 200:
        sha = current.json().get("sha")
    elif current.status_code != 404:
        raise RuntimeError(f"GitHub檢查舊檔失敗：HTTP {current.status_code} {current.text[:300]}")
    body = {
        "message": commit_message[:120],
        "content": base64.b64encode(contents).decode("ascii"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    response = requests.put(url, headers=headers, json=body, timeout=60)
    if not response.ok:
        raise RuntimeError(f"GitHub儲存檔案失敗：HTTP {response.status_code} {response.text[:500]}")
    result = response.json()
    return {
        "path": str(result.get("content", {}).get("path", path)),
        "sha": str(result.get("content", {}).get("sha", "")),
        "html_url": str(result.get("content", {}).get("html_url", "")),
        "commit_sha": str(result.get("commit", {}).get("sha", "")),
    }
