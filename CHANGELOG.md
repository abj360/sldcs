# Changelog

All notable changes to SLDCS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
semantic versioning once a first release is tagged.

## Unreleased

### Added
- Project setup script that reproducibly creates runtime directories and registry/dataset skeletons.
- Google Drive sync script for resumable import of the raw specimen images.
- Raw dataset validation with a corrupted/duplicate/undersized quality report.
- Dataset preparation pipeline that produces the flat annotation-ready image and label sets, a manifest, a summary, and an off-tree raw backup.
- Annotation guide generator, including worked YOLO-format examples, placeholder diagrams, a label-format validator, and a QA checklist.
