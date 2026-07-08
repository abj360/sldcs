# Raw specimen images

This directory holds the raw, unmodified shrimp larvae specimen photographs that
every downstream stage of SLDCS depends on. It is the immutable source of truth
for the dataset.

## Where the data comes from

The images originate from the UAPB Aquaculture Research Program's shared Google
Drive folder:

> **Drive folder:** `<FILL IN: paste the Google Drive folder link or ID here>`

The expected contents are **250 raw specimen images**.

## How the data gets here

This directory is **never hand-edited and never populated manually.** It is
written to by exactly one program:

```bash
python scripts/google_drive_sync.py --folder-id <DRIVE_FOLDER_ID>
```

(The folder ID and credentials path can also be supplied through the
`GOOGLE_DRIVE_FOLDER_ID` and `GOOGLE_DRIVE_CREDENTIALS_PATH` environment
variables; see `.env.example`.) The sync is resumable: re-running it only fetches
images that are missing or incomplete, and it writes a `sync_report.json` here
recording what was downloaded, skipped, or failed.

Keeping a single automated writer means the contents of this directory are
always traceable back to one source: the Drive folder above. If a file is here,
it came from that folder through that script — nothing else.

## What must not happen here

- Do not add, rename, edit, or delete images by hand.
- Do not annotate here. Annotation happens on the processed copy produced by
  `scripts/prepare_dataset.py`.
- Do not treat anything here as output; this directory is input only.
