# Anonymization Notes

Applied automatic anonymization in copied text files:
- Local absolute paths replaced with placeholders (`<PROJECT_ROOT>`, `<WORKSPACE_ROOT>`).
- API keys redacted (`sk-REDACTED`).
- Provider domains normalized to generic placeholders (`provider-a.example`, `provider-b.example`).
- User-like identifiers replaced (`anonymous_user`).
- Method alias normalized to `geoskill_skill_graph` where needed for anonymous review.

Excluded from this package:
- third-party comparison adapter code
- external comparison runner config blocks
