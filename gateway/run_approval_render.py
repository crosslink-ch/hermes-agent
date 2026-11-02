"""Text approval rendering for surfaces without interactive cards."""
from __future__ import annotations

def _format_exec_approval_fallback(
    command: str, description: str, command_prefix: str, *, allow_permanent: bool = True,
    allow_session: bool = True, smart_denied: bool = False) -> str:
    """Render the text fallback from approval capabilities, not platform names. Same words as
    the button card (``BasePlatformAdapter._format_exec_approval``), plus the typed ``/approve``
    steps a surface without buttons needs. The command is never cut: ``/approve`` must only ever
    answer a command the user could read in full (#62359)."""
    from gateway.run import t
    from gateway.platforms.base_exec_approval import (
        approval_timeout_seconds, ea_header_text, ea_reason_label_text, format_approval_deadline_line)
    heading = (t("gateway.exec_approval.smart_deny_heading") if smart_denied
               else f"⚠️ **{ea_header_text()}**")

    choices = [t("gateway.exec_approval.text_choice_once", prefix=command_prefix)]
    if not smart_denied and allow_session:
        choices.append(t("gateway.exec_approval.text_choice_session", prefix=command_prefix))
        if allow_permanent:
            choices.append(t("gateway.exec_approval.text_choice_always", prefix=command_prefix))
    choices.append(t("gateway.exec_approval.text_choice_deny", prefix=command_prefix))
    return (
        t("gateway.exec_approval.text_body", heading=heading, command=command,  # whole: send() splits it
          reason_label=ea_reason_label_text(), reason=description)
        + t("gateway.exec_approval.text_choice_joiner").join(choices[:-1])
        + t("gateway.exec_approval.text_choice_last", choice=choices[-1])
        + format_approval_deadline_line(approval_timeout_seconds()))

