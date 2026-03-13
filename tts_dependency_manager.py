#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as installed_version

try:
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover - optional dependency
    InvalidVersion = Exception
    Version = None


TRACKED_TTS_PACKAGES = ("edge-tts",)


@dataclass(frozen=True)
class PackageVersionStatus:
    distribution: str
    installed: str | None
    latest: str | None
    needs_update: bool
    error: str | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _version_key(version_text: str) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"[.\-+_]+", version_text)
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _is_newer(installed: str | None, latest: str | None) -> bool:
    if not latest:
        return False
    if not installed:
        return True
    if Version is not None:
        try:
            return Version(latest) > Version(installed)
        except InvalidVersion:
            pass
    return _version_key(latest) > _version_key(installed)


def get_installed_distribution_version(distribution: str) -> str | None:
    try:
        return installed_version(distribution)
    except PackageNotFoundError:
        return None


def fetch_latest_pypi_version(distribution: str, timeout: float = 5.0) -> str:
    url = f"https://pypi.org/pypi/{distribution}/json"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    info = payload.get("info") or {}
    version_value = str(info.get("version") or "").strip()
    if not version_value:
        raise RuntimeError(f"PyPI returned no version for {distribution}")
    return version_value


def inspect_tts_dependencies() -> list[PackageVersionStatus]:
    statuses: list[PackageVersionStatus] = []
    for distribution in TRACKED_TTS_PACKAGES:
        current = get_installed_distribution_version(distribution)
        try:
            latest = fetch_latest_pypi_version(distribution)
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            statuses.append(
                PackageVersionStatus(
                    distribution=distribution,
                    installed=current,
                    latest=None,
                    needs_update=False,
                    error=str(exc),
                )
            )
            continue

        statuses.append(
            PackageVersionStatus(
                distribution=distribution,
                installed=current,
                latest=latest,
                needs_update=_is_newer(current, latest),
            )
        )
    return statuses


def _format_status(status: PackageVersionStatus) -> str:
    if status.error:
        return (
            f"[tts-sync] {status.distribution}: не удалось проверить обновления"
            f" ({status.error})"
        )
    if status.needs_update:
        current = status.installed or "not installed"
        return (
            f"[tts-sync] {status.distribution}: доступно обновление "
            f"{current} -> {status.latest}"
        )
    version_text = status.installed or status.latest or "unknown"
    return f"[tts-sync] {status.distribution}: актуально ({version_text})"


def sync_tts_dependencies(*, apply_updates: bool = False) -> list[PackageVersionStatus]:
    statuses = inspect_tts_dependencies()
    for status in statuses:
        print(_format_status(status))
        if not apply_updates or not status.needs_update or not status.latest:
            continue

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    f"{status.distribution}=={status.latest}",
                ],
                check=True,
            )
            print(
                f"[tts-sync] {status.distribution}: обновлено до {status.latest}"
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[tts-sync] {status.distribution}: автообновление не удалось ({exc})"
            )
    return statuses


def sync_tts_dependencies_from_env() -> list[PackageVersionStatus]:
    if _env_flag("AUDIOBOOKER_SKIP_TTS_SYNC", False):
        return []

    should_check = _env_flag("CHECK_TTS_DEPS_ON_START", True)
    apply_updates = _env_flag("AUTO_UPDATE_TTS_DEPS", False)
    if not should_check and not apply_updates:
        return []
    return sync_tts_dependencies(apply_updates=apply_updates)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or update runtime TTS package versions."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Install the latest tracked TTS package versions from PyPI.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sync_tts_dependencies(apply_updates=args.apply)


if __name__ == "__main__":
    main()
