"""Shared HTTP, cache and disk-guard helpers. Standard library only."""
import gzip
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"

USER_AGENT = "classical-form-dataset-builder/0.1 (student research script)"
MIN_INTERVAL = 1.0  # seconds between requests, across all hosts
WIKIMEDIA_FILE_HOST = "upload.wikimedia.org"
WIKIMEDIA_FILE_INTERVAL = 6.0  # seconds; this host rate-limited us at 1 request/second
RATE_LIMIT_WAIT = 60  # first wait after a 429, doubled on each retry
MAX_ATTEMPTS = 5

# Phase 2 safety limits (approved cap): stop rather than fill the disk.
MAX_DATA_BYTES = 12_000_000_000
MIN_FREE_BYTES = 8_000_000_000

_last_request = 0.0
_requests_since_check = 0
_last_wikimedia_file = -WIKIMEDIA_FILE_INTERVAL


class DiskGuardError(RuntimeError):
    pass


class RateLimitedError(Exception):
    """A server kept answering 429: the run should stop instead of trying other files."""


def data_size():
    total = 0
    for path in DATA.rglob("*"):
        try:  # Other jobs delete temporary files while this scan runs.
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def check_disk():
    DATA.mkdir(parents=True, exist_ok=True)
    used = data_size()
    free = shutil.disk_usage(DATA).free
    if used > MAX_DATA_BYTES:
        raise DiskGuardError(f"data/ is {used/1e6:.0f} MB, over the {MAX_DATA_BYTES/1e6:.0f} MB limit")
    if free < MIN_FREE_BYTES:
        raise DiskGuardError(f"only {free/1e9:.1f} GB free, under the {MIN_FREE_BYTES/1e9:.0f} GB floor")
    return used, free


def _throttle():
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _guard_periodically():
    global _requests_since_check
    _requests_since_check += 1
    if _requests_since_check >= 50:
        _requests_since_check = 0
        check_disk()


def http_get(url):
    """GET with throttling and retries. Returns bytes, or None on a permanent 404/410."""
    _guard_periodically()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        _throttle()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (404, 410):
                return None
            last_error = error
            retry_after = error.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
            if error.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            delay = 2 ** (attempt + 1)
        time.sleep(min(delay, 120))
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts: {url}: {last_error}")


def cached_json(url, namespace):
    """Fetch JSON through a gzip cache under data/cache/<namespace>. Returns None on 404."""
    key = hashlib.sha256(url.encode()).hexdigest()
    path = CACHE / namespace / key[:2] / f"{key}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    body = http_get(url)
    if body is None:
        return None
    parsed = json.loads(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(parsed, handle)
    temporary.replace(path)
    return parsed


def download_file(url, destination):
    """Download once to a path, skipping it if already present."""
    destination = Path(destination)
    if destination.exists():
        return False
    body = http_get(url)
    if body is None:
        raise RuntimeError(f"not found: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(destination)
    return True


def quote_url(url):
    """Percent-encode a URL's path so unusual file names can be requested."""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_host(url):
    """Space out requests to Wikimedia's file server more than other hosts."""
    global _last_wikimedia_file
    if urllib.parse.urlsplit(url).netloc == WIKIMEDIA_FILE_HOST:
        remaining = WIKIMEDIA_FILE_INTERVAL - (time.monotonic() - _last_wikimedia_file)
        if remaining > 0:
            time.sleep(remaining)
        _last_wikimedia_file = time.monotonic()


def download_stream(url, destination, max_bytes):
    """Stream one file to disk, throttled and retried. Existing files are kept, not re-fetched."""
    destination = Path(destination)
    if destination.exists():
        return sha256_file(destination), destination.stat().st_size, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(quote_url(url), headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        _throttle()
        wait_for_host(url)
        try:
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    size += len(chunk)
                    if size > max_bytes:
                        raise DiskGuardError(f"over the {max_bytes/1e6:.0f} MB per-file limit: {url}")
                    digest.update(chunk)
                    handle.write(chunk)
            partial.replace(destination)
            return digest.hexdigest(), size, False
        except urllib.error.HTTPError as error:
            partial.unlink(missing_ok=True)
            if error.code not in (429, 500, 502, 503, 504):
                raise
            last_error = error
            retry_after = error.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
            if error.code == 429:
                delay = max(delay, RATE_LIMIT_WAIT * 2 ** attempt)
                if attempt == MAX_ATTEMPTS - 1:
                    raise RateLimitedError(f"still rate-limited after {MAX_ATTEMPTS} attempts: {url}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            partial.unlink(missing_ok=True)
            last_error = error
            delay = 2 ** (attempt + 1)
        except DiskGuardError:
            partial.unlink(missing_ok=True)
            raise
        time.sleep(min(delay, 960))
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts: {url}: {last_error}")
