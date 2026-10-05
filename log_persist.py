#!/usr/bin/env python3
"""GitHub-backed persistence for logged picks.

Render's free tier uses an ephemeral filesystem — every spin-down (after
15 min of inactivity) and every deploy wipes `logs/`. That broke the
daily-picks history: snapshots written today would vanish before tomorrow.

This module mirrors `logs/*.json` to the GitHub repo via the Contents API,
so the logs survive restarts. Everything is OPTIONAL — if GITHUB_TOKEN
isn't set, the module silently no-ops and plays_log falls back to local-only
storage.

Setup on Render (one-time):
  1. Create a fine-grained PAT at github.com/settings/tokens scoped to
     only this repo with Contents: Read & Write permission.
  2. Add it as the GITHUB_TOKEN environment variable on the Render service.
  3. (Optional) Set LOGS_REPO=owner/repo if different from the default,
     and LOGS_BRANCH if different from "main". The default repo path
     where logs are stored is "logs/" in the main branch.
"""
import base64
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "MauricioM1103/first-pitch"
# Pick snapshots used to land on `main`, which caused every save (sometimes
# every few minutes while the app was being viewed) to trigger a fresh Render
# auto-deploy and burn pipeline minutes. The data branch is code-free, so
# commits here never trigger a build. Override with LOGS_BRANCH if you want
# the old behavior.
DEFAULT_BRANCH = "data"
DEFAULT_LOGS_PATH = "logs"
_BRANCH_CREATION_CHECKED = False

# Per-process in-memory cache for file SHAs so a save doesn't need an extra
# GET before each PUT. Cleared on process restart (which is fine).
_SHA_CACHE = {}


def _token():
    return os.environ.get("GITHUB_TOKEN")


def _repo_slug():
    return os.environ.get("LOGS_REPO", DEFAULT_REPO)


def _branch():
    return os.environ.get("LOGS_BRANCH", DEFAULT_BRANCH)


def is_configured():
    """True if we have a token and should use GitHub as the source of truth."""
    return bool(_token())


def _api(url, method="GET", data=None, timeout=20):
    token = _token()
    if not token:
        return None
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = Request(url, method=method, data=body)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "betting-tools-logs")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else None
    except HTTPError as e:
        if e.code == 404:
            return None
        # Surface auth / rate-limit errors to the caller via exception
        raise


def read_file(path):
    """Read a UTF-8 text file from the repo. Returns (text, sha) or (None, None)."""
    if not is_configured():
        return None, None
    url = f"{GITHUB_API}/repos/{_repo_slug()}/contents/{path}?ref={_branch()}"
    try:
        data = _api(url)
    except Exception:
        return None, None
    if not data or data.get("type") != "file":
        return None, None
    sha = data.get("sha")
    _SHA_CACHE[path] = sha
    content_b64 = data.get("content", "") or ""
    try:
        text = base64.b64decode(content_b64).decode("utf-8")
    except Exception:
        return None, None
    return text, sha


def _ensure_branch_exists():
    """Create the target branch if it doesn't exist yet. Idempotent — the
    GitHub API call checks once per process. First call does one GET + one
    POST, every subsequent call is a cached no-op."""
    global _BRANCH_CREATION_CHECKED
    if _BRANCH_CREATION_CHECKED or not is_configured():
        return
    _BRANCH_CREATION_CHECKED = True
    branch = _branch()
    repo = _repo_slug()
    # Does the branch already exist?
    try:
        _api(f"{GITHUB_API}/repos/{repo}/git/refs/heads/{branch}")
        return  # exists
    except HTTPError as e:
        if e.code != 404:
            return  # some other error — don't try to create
    except Exception:
        return
    # Branch doesn't exist — fork from main's current head.
    try:
        main_ref = _api(f"{GITHUB_API}/repos/{repo}/git/refs/heads/main")
        main_sha = (main_ref or {}).get("object", {}).get("sha")
        if not main_sha:
            return
        _api(
            f"{GITHUB_API}/repos/{repo}/git/refs",
            method="POST",
            data={"ref": f"refs/heads/{branch}", "sha": main_sha},
        )
    except Exception:
        pass


def write_file(path, text, message=None):
    """Write (create or update) a UTF-8 text file in the repo.
    Returns True on success, False otherwise (silently — never raises)."""
    if not is_configured():
        return False
    _ensure_branch_exists()
    url = f"{GITHUB_API}/repos/{_repo_slug()}/contents/{path}"
    # Need the current sha to update an existing file. Try cache, then GET.
    sha = _SHA_CACHE.get(path)
    if sha is None:
        _, sha = read_file(path)
    body = {
        "message": message or f"log: update {path}",
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "branch":  _branch(),
    }
    if sha:
        body["sha"] = sha
    try:
        resp = _api(url, method="PUT", data=body)
        # Save new sha for next write
        if resp and isinstance(resp, dict):
            new_sha = (resp.get("content") or {}).get("sha")
            if new_sha:
                _SHA_CACHE[path] = new_sha
        return True
    except HTTPError as e:
        # 409: SHA mismatch — stale cache. Refetch sha and retry once.
        if e.code == 409:
            _SHA_CACHE.pop(path, None)
            _, sha = read_file(path)
            if sha:
                body["sha"] = sha
                try:
                    resp = _api(url, method="PUT", data=body)
                    if resp and isinstance(resp, dict):
                        new_sha = (resp.get("content") or {}).get("sha")
                        if new_sha:
                            _SHA_CACHE[path] = new_sha
                    return True
                except Exception:
                    return False
        return False
    except Exception:
        return False


_LIST_CACHE = {}
_LIST_TTL_S = 60


def list_dir(path):
    """List files under a repo path. Returns [{name, path, sha, type}].
    Cached per-process for 60s to avoid a GitHub API hit on every /logged load.
    """
    if not is_configured():
        return []
    now = time.time()
    hit = _LIST_CACHE.get(path)
    if hit and now - hit[1] < _LIST_TTL_S:
        return hit[0]
    url = f"{GITHUB_API}/repos/{_repo_slug()}/contents/{path}?ref={_branch()}"
    try:
        data = _api(url)
    except Exception:
        return hit[0] if hit else []
    out = data if isinstance(data, list) else []
    _LIST_CACHE[path] = (out, now)
    return out


def invalidate_list_cache():
    _LIST_CACHE.clear()


def status_summary():
    """Return a dict describing GitHub persistence state for the /logged page."""
    if not is_configured():
        return {
            "enabled": False,
            "reason": "GITHUB_TOKEN env var not set — logs are stored only on "
                      "this instance's local disk and will be lost on restart.",
        }
    try:
        files = list_dir(DEFAULT_LOGS_PATH)
    except Exception as e:
        return {"enabled": True, "error": f"GitHub API reachable but errored: {e}"}
    picks_files = sum(1 for f in files if isinstance(f, dict)
                      and f.get("name", "").startswith("picks_"))
    graded_files = sum(1 for f in files if isinstance(f, dict)
                       and f.get("name", "").startswith("graded_"))
    return {
        "enabled": True,
        "repo":    _repo_slug(),
        "branch":  _branch(),
        "picks_files":  picks_files,
        "graded_files": graded_files,
    }


if __name__ == "__main__":
    print(json.dumps(status_summary(), indent=2))
