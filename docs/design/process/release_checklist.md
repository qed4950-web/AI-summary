# Release Checklist

- [ ] Run `scripts/dev/setup_env.sh`
- [ ] Execute smoke tests (scan/train/chat/watch) on staging dataset
- [ ] Capture KPI snapshot: `python scripts/util/release_prepare.py`
- [ ] Verify meeting/photo agents basic pipelines run
- [ ] Update `docs/cycles/` logs and release notes
- [ ] Tag release and upload build artifacts (PyInstaller, configs)
- [ ] Share audit log summary with stakeholders
