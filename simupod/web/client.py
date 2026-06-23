"""Minimal HTTP client over ``urllib`` (no third-party dependency).

Bearer-authenticated JSON requests with capped-backoff retry on 5xx/network
errors (idempotent GETs). 4xx errors raise immediately as ``WebError`` carrying
the parsed ``detail``. ``urllib`` transparently follows the 302 that the result
endpoint returns in prod (to a signed object-store URL).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .config import WebConfig, WebError

_RETRIES = 3


def _parse_detail(body: str) -> Any:
    try:
        obj = json.loads(body)
        return obj.get("detail", obj) if isinstance(obj, dict) else obj
    except Exception:
        return body


class HttpClient:
    def __init__(self, cfg: WebConfig):
        self.cfg = cfg

    def _open(self, method: str, path: str, *, body: Optional[dict] = None):
        url = self.cfg.url + path
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)

        last: Optional[Exception] = None
        for attempt in range(_RETRIES):
            try:
                return urllib.request.urlopen(req, timeout=self.cfg.request_timeout_s)
            except urllib.error.HTTPError as e:
                detail = _parse_detail(e.read().decode(errors="replace"))
                if 500 <= e.code < 600 and attempt < _RETRIES - 1:
                    last = e
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise WebError(f"{method} {path} -> HTTP {e.code}",
                               status_code=e.code, body=detail)
            except urllib.error.URLError as e:
                last = e
                if attempt < _RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise WebError(f"{method} {path} failed: {e.reason}")
        raise WebError(f"{method} {path} failed after retries: {last}")

    def get_json(self, path: str) -> dict:
        with self._open("GET", path) as r:
            return json.load(r)

    def post_json(self, path: str, body: dict) -> dict:
        with self._open("POST", path, body=body) as r:
            return json.load(r)

    def delete(self, path: str) -> int:
        with self._open("DELETE", path) as r:
            return r.status

    def get_bytes(self, path: str) -> bytes:
        with self._open("GET", path) as r:
            return r.read()

    # --- typed endpoint helpers -------------------------------------------

    def whoami(self) -> dict:
        return self.get_json("/v1/auth/whoami")

    def account(self) -> dict:
        return self.get_json("/v1/account")

    def estimate(self, spec: dict) -> dict:
        return self.post_json("/v1/estimate", {"spec": spec})

    def create_api_key(self, name: str = "default") -> dict:
        return self.post_json("/v1/keys", {"name": name})

    def submit_job(self, spec: dict, *, name=None, device=None,
                   quote_id=None) -> dict:
        body: dict = {"spec": spec}
        if name is not None:
            body["name"] = name
        if device is not None:
            body["device"] = device
        if quote_id is not None:
            body["quote_id"] = quote_id
        return self.post_json("/v1/jobs", body)

    def get_job(self, job_id: str) -> dict:
        return self.get_json(f"/v1/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict:
        return self.post_json(f"/v1/jobs/{job_id}/cancel", {})

    def download_result(self, job_id: str) -> bytes:
        return self.get_bytes(f"/v1/jobs/{job_id}/result")
