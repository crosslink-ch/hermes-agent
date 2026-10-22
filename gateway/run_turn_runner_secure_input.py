"""v3 prompt callbacks owned by one admitted TheChat executor turn."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from gateway.run_turn_runner_vault import _publish


class TurnSecureInput:
    def __init__(self, turn, adapter, context, profile_id, home, allowed, owner):
        self.turn, self.adapter, self.context = turn, adapter, context
        self.profile_id, self.home = profile_id, home
        self.allowed, self.owner = allowed, owner
        raw = None
        if allowed and adapter._secure_input_supported and owner == turn._ctx.source.user_id:
            from gateway.config import Platform, load_gateway_config
            platform = load_gateway_config().platforms.get(Platform.THECHAT)
            raw = platform.extra.get("secure_input") if platform else None
        self.opt_ins = raw if isinstance(raw, dict) else {}
        self.active = threading.Event()
        self.active.set()
        self.records = []

    def authorized(self, kind):
        """Transport-thread-safe authority: never consult that thread's profile scope."""
        ctx, adapter = self.turn._ctx, self.adapter
        return bool(self.active.is_set() and self.allowed and adapter is not None
                    and adapter._secure_input_supported and adapter.webhook_url
                    and adapter._client is not None and not adapter._secure_input.closed
                    and ctx._run_still_current() and not self.turn._agent_interrupted()
                    and adapter._event_contexts.get(str(ctx.source.message_id or "")) == self.context
                    and (kind != "sudo.password" or self.opt_ins.get("allow_sudo") is True)
                    and (kind != "skill.secret" or self.opt_ins.get("allow_skill_secrets") is True)
                    and (kind == "vault.code" or (self.owner and self.owner == ctx.source.user_id
                                                  and adapter._owner_user_id == self.owner)))

    def live(self, kind):
        from hermes_constants import get_hermes_home
        from agent.vault_backends.unlock import unlock_attempt_is_current
        return bool(self.authorized(kind) and str(get_hermes_home()) == self.home
                    and (kind != "onepassword.unlock" or unlock_attempt_is_current("onepassword")))

    def request(self, kind, target):
        if not self.live(kind):
            return None
        broker = self.adapter._secure_input
        record = broker.create(context=self.context, profile_id=self.profile_id,
                               session_key=self.turn._ctx.session_key, prompt_kind=kind, target=target,
                               authorized=lambda: self.authorized(kind))
        self.records.append(record)
        future = None
        answer = None
        try:
            future = _publish(self.turn, self.adapter, self.context, {
                "type": "secure.input.request", "status": "waiting", "label": "Secure input",
                "payload": dict(record.payload)}, retry=True)
            if future is None:
                broker.cancel(record, outcome="failed")
            while not record.event.wait(0.1):
                if not self.live(kind):
                    broker.cancel(record)
                elif record.deadline <= time.monotonic() or record.payload["expiresAt"] <= int(time.time() * 1000):
                    broker.cancel(record, outcome="expired")
                elif future is not None and future.done():
                    try:
                        if not future.result().success:
                            broker.cancel(record, outcome="failed")
                    except Exception:
                        broker.cancel(record, outcome="failed")
            if not self.live(kind):
                broker.cancel(record)
            elif record.outcome == "failed":
                raise RuntimeError("Secure input prompt could not be delivered")
            else:
                answer = record.take_response()
        finally:
            if record.outcome is None:
                broker.cancel(record)
            if future is not None and not future.done():
                future.cancel()
            # This is the server's post-answer authorization fence, not merely
            # best-effort UI cleanup. A cached owner cannot override a denial.
            resolved = None
            resolution_accepted = False
            try:
                if self.adapter._client is not None:
                    resolved = _publish(self.turn, self.adapter, self.context, {
                        "type": "secure.input.resolved", "status": "completed", "label": "Secure input closed",
                        "payload": {"version": 3, "requestId": record.payload["requestId"],
                                    "sessionKey": record.payload["sessionKey"], "outcome": record.outcome}
                    })
                if resolved is not None:
                    resolution_accepted = resolved.result(timeout=3).success is True
            except Exception:
                if resolved is not None:
                    resolved.cancel()
            finally:
                if not resolution_accepted and answer is not None:
                    answer.clear()
                broker.release(record)
        # Resolution publication is also an await point: revocation/finalization may win there.
        if not resolution_accepted or not self.live(kind):
            if answer:
                answer.clear()
            return None
        return answer

    def code(self, site, _hint):
        return (self.request("vault.code", site) or {}).get("code", "")

    def save_login(self, origin, _site):
        return self.request("vault.save_login", origin)

    def sudo(self):
        # Commands can contain unknown credentials; the fixed description cannot leak them.
        return (self.request("sudo.password", "Privileged terminal command") or {}).get("password", "")

    def skill_secret(self, name, _prompt, _metadata):
        from hermes_cli.config import save_env_value_secure
        answer = self.request("skill.secret", name)
        try:
            if not answer or not self.live("skill.secret"):
                return {"success": False, "skipped": True}
            return {**save_env_value_secure(name, answer["value"], preserve_exact=True), "skipped": False}
        except Exception:
            return {"success": False, "skipped": True}
        finally:
            if answer:
                answer.clear()

    def unlock(self, backend, _display):
        return (self.request("onepassword.unlock", "1Password") or {}).get("password", "") if backend == "onepassword" else ""

    def close(self):
        self.active.clear()
        for record in self.records:
            self.adapter._secure_input.release(record)
        self.records.clear()


@contextmanager
def secure_input_turn_scope(turn, adapter, context, profile_id, home, allowed, owner):
    from agent.vault_backends import unlock
    from tools.skills_tool_setup import secret_capture_scope
    from tools.terminal_tool_sudo import sudo_prompt_scope
    prompts = TurnSecureInput(turn, adapter, context, profile_id, home, allowed, owner)
    previous = (unlock.get_code_prompt_callback(), unlock.get_save_login_prompt_callback(), unlock.get_unlock_prompt_callback())
    def unlock_prompt(backend, display):
        if backend == "bitwarden":
            return previous[2](backend, display) if previous[2] else ""
        return prompts.unlock(backend, display)
    # Even a denied turn must override a callback left on a reused Desktop worker.
    unlock.set_code_prompt_callback(prompts.code if prompts.authorized("vault.code") else None)
    unlock.set_save_login_prompt_callback(prompts.save_login if prompts.authorized("vault.save_login") else None)
    unlock.set_unlock_prompt_callback(unlock_prompt if previous[2] or prompts.authorized("onepassword.unlock") else None)
    backends = ({"bitwarden"} if previous[2] else set()) | ({"onepassword"} if prompts.authorized("onepassword.unlock") else set())
    try:
        with unlock.native_prompt_scope(backends), secret_capture_scope(prompts.skill_secret if prompts.authorized("skill.secret") else None), \
                sudo_prompt_scope(prompts.sudo if prompts.authorized("sudo.password") else None,
                                  is_current=lambda: prompts.live("sudo.password")):
            yield
    finally:
        prompts.close()
        unlock.set_code_prompt_callback(previous[0])
        unlock.set_save_login_prompt_callback(previous[1])
        unlock.set_unlock_prompt_callback(previous[2])
