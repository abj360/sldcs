"""Import the raw specimen images from Google Drive, resumably.

This script is the single automated entry point for pulling the project's raw
specimen photographs out of the shared Google Drive folder into ``data/raw``. It
authenticates with a service-account (or OAuth) credentials file, lists the
image files in the target folder, and downloads any that are not already present
locally with a matching byte size. Because the download step skips files that
already exist intact, the whole operation is safe to re-run after a partial or
failed sync without re-fetching everything.

This script performs I/O only; it does not annotate, validate, or transform the
images it downloads.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Read-only Drive access is sufficient; never request write scope.
DRIVE_SCOPES: Final[list[str]] = ["https://www.googleapis.com/auth/drive.readonly"]

# Fields requested per file. Kept minimal and explicit so the returned dicts
# always contain exactly the keys downstream code relies on.
FILE_FIELDS: Final[str] = "nextPageToken, files(id, name, size, mimeType)"

DEFAULT_DESTINATION: Final[str] = "data/raw"
DEFAULT_REPORT_NAME: Final[str] = "sync_report.json"
DOWNLOAD_CHUNK_SIZE_BYTES: Final[int] = 5 * 1024 * 1024


class DriveAuthenticationError(RuntimeError):
    """Raised when Google Drive authentication cannot be completed.

    Carries a message that names the exact environment variable or file path the
    human must fix, so a configuration mistake never surfaces as an opaque
    library traceback.
    """


def authenticate_google_drive(credentials_path: str) -> Resource:
    """Build an authenticated Google Drive API service object.

    Args:
        credentials_path: Filesystem path to a service-account (or OAuth)
            credentials JSON file.

    Returns:
        An authenticated Drive ``Resource`` ready to issue API calls.

    Raises:
        DriveAuthenticationError: If the credentials file is missing or cannot be
            used to authenticate, with a message telling the operator exactly
            what to fix.
    """
    path = Path(credentials_path)
    if not path.is_file():
        raise DriveAuthenticationError(
            f"Google Drive credentials not found at '{credentials_path}'. "
            "Set GOOGLE_DRIVE_CREDENTIALS_PATH (see .env.example) to a valid "
            "service-account JSON file."
        )
    try:
        credentials = Credentials.from_service_account_file(
            str(path), scopes=DRIVE_SCOPES
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except Exception as error:  # noqa: BLE001 - re-raised as a domain error below
        raise DriveAuthenticationError(
            f"Failed to authenticate with Google Drive using '{credentials_path}': "
            f"{error}. Confirm the file is a valid service-account key and that "
            "the Drive API is enabled for its project."
        ) from error


def list_files_in_folder(service: Resource, folder_id: str) -> list[dict]:
    """List every image file directly contained in a Drive folder.

    Pages through the Drive API so folders larger than a single response page are
    fully enumerated, and restricts results to image MIME types.

    Args:
        service: Authenticated Drive ``Resource`` from
            :func:`authenticate_google_drive`.
        folder_id: ID of the Drive folder to enumerate.

    Returns:
        A list of dicts, each with at least ``id``, ``name``, ``size``, and
        ``mimeType`` keys, for every image file in the folder.

    Raises:
        HttpError: If the Drive API rejects the query.
    """
    query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
    files: list[dict] = []
    page_token: str | None = None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields=FILE_FIELDS,
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def _download_single_file(service: Resource, file_id: str, destination: Path) -> None:
    """Stream one Drive file's bytes to a local path.

    Args:
        service: Authenticated Drive ``Resource``.
        file_id: Drive file ID to download.
        destination: Local path to write the file to.
    """
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(str(destination), mode="wb") as file_handle:
        downloader = MediaIoBaseDownload(
            file_handle, request, chunksize=DOWNLOAD_CHUNK_SIZE_BYTES
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()


def download_images(
    service: Resource, files: list[dict], destination_dir: Path
) -> dict:
    """Download image files into a destination directory, skipping intact ones.

    A file is treated as already downloaded when a local file of the same name
    exists with a byte size matching the Drive metadata; this is what makes the
    sync resumable. A failure on one file is recorded and does not abort the
    remaining downloads.

    Args:
        service: Authenticated Drive ``Resource``.
        files: File metadata dicts as returned by :func:`list_files_in_folder`.
        destination_dir: Local directory to download into (created if absent).

    Returns:
        A summary dict ``{"downloaded": int, "skipped": int, "failed": list[str]}``
        where each ``failed`` entry is ``"<name>: <error>"``.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for file_meta in files:
        name = file_meta["name"]
        target = destination_dir / name
        expected_size = int(file_meta["size"]) if file_meta.get("size") else None
        if target.exists() and expected_size is not None and target.stat().st_size == expected_size:
            skipped += 1
            continue
        try:
            _download_single_file(service, file_meta["id"], target)
            downloaded += 1
        except (HttpError, OSError) as error:
            if target.exists():
                target.unlink(missing_ok=True)
            failed.append(f"{name}: {error}")

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def generate_download_report(summary: dict, report_path: Path) -> None:
    """Write the sync summary plus a UTC timestamp to a JSON report.

    Args:
        summary: The dict returned by :func:`download_images`.
        report_path: Destination path for the JSON report.
    """
    report = dict(summary)
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full raw-image sync from the command line.

    Reads the folder ID and credentials path from the environment (overridable
    via ``--folder-id`` and ``--credentials``), downloads all images into the
    destination directory, writes a sync report, and prints the final counts.

    Raises:
        SystemExit: If no Drive folder ID is configured.
    """
    parser = argparse.ArgumentParser(description="Sync raw specimen images from Google Drive.")
    parser.add_argument(
        "--folder-id",
        default=os.environ.get("GOOGLE_DRIVE_FOLDER_ID", ""),
        help="Google Drive folder ID (defaults to $GOOGLE_DRIVE_FOLDER_ID).",
    )
    parser.add_argument(
        "--destination",
        default=DEFAULT_DESTINATION,
        help=f"Local download directory (default: {DEFAULT_DESTINATION}).",
    )
    parser.add_argument(
        "--credentials",
        default=os.environ.get(
            "GOOGLE_DRIVE_CREDENTIALS_PATH", "credentials/service_account.json"
        ),
        help="Path to service-account credentials JSON "
        "(defaults to $GOOGLE_DRIVE_CREDENTIALS_PATH).",
    )
    args = parser.parse_args()

    if not args.folder_id:
        raise SystemExit(
            "No Drive folder ID provided. Pass --folder-id or set "
            "GOOGLE_DRIVE_FOLDER_ID (see .env.example)."
        )

    destination_dir = Path(args.destination)
    service = authenticate_google_drive(args.credentials)
    files = list_files_in_folder(service, args.folder_id)
    print(f"Found {len(files)} image file(s) in Drive folder {args.folder_id}.")

    summary = download_images(service, files, destination_dir)
    generate_download_report(summary, destination_dir / DEFAULT_REPORT_NAME)

    print(
        f"Sync complete: {summary['downloaded']} downloaded, "
        f"{summary['skipped']} skipped, {len(summary['failed'])} failed."
    )
    for failure in summary["failed"]:
        print(f"  ! {failure}")


if __name__ == "__main__":
    main()
