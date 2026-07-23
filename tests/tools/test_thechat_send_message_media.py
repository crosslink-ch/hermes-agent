from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"


class _FakeTheChatAdapter:
    instances = []

    def __init__(self, _config):
        self.calls = []
        self.__class__.instances.append(self)

    async def connect_outbound(self):
        self.calls.append(("connect",))
        return True

    async def disconnect(self):
        self.calls.append(("disconnect",))

    async def send(self, *, chat_id, content, metadata=None):
        self.calls.append(("send", chat_id, content, metadata))
        return SimpleNamespace(success=True, message_id="text-1", error=None)

    async def send_image_file(self, *, chat_id, image_path, caption=None, metadata=None):
        self.calls.append(("image", chat_id, image_path, caption, metadata))
        return SimpleNamespace(success=True, message_id="image-1", error=None)

    async def send_document(self, *, chat_id, file_path, caption=None, metadata=None):
        self.calls.append(("document", chat_id, file_path, caption, metadata))
        return SimpleNamespace(success=True, message_id="document-1", error=None)

    async def send_voice(self, *, chat_id, audio_path, caption=None, metadata=None):
        self.calls.append(("voice", chat_id, audio_path, caption, metadata))
        return SimpleNamespace(success=True, message_id="voice-1", error=None)

    async def send_video(self, *, chat_id, video_path, caption=None, metadata=None):
        self.calls.append(("video", chat_id, video_path, caption, metadata))
        return SimpleNamespace(success=True, message_id="video-1", error=None)


@pytest.fixture(autouse=True)
def _patch_ephemeral_adapter(monkeypatch):
    _FakeTheChatAdapter.instances.clear()
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    monkeypatch.setattr(
        "gateway.platforms.thechat.TheChatAdapter",
        _FakeTheChatAdapter,
    )


@pytest.mark.asyncio
async def test_proactive_thechat_image_uses_outbound_attachment_adapter():
    result = await _send_to_platform(
        Platform.THECHAT,
        SimpleNamespace(token="token", extra={"base_url": "https://thechat.test"}),
        CONVERSATION_ID,
        "image caption",
        thread_id="thread-1",
        media_files=[("/safe/plot.png", False)],
    )

    assert result == {"success": True, "message_id": "image-1"}
    assert _FakeTheChatAdapter.instances[0].calls == [
        ("connect",),
        (
            "image",
            CONVERSATION_ID,
            "/safe/plot.png",
            "image caption",
            {"thread_id": "thread-1"},
        ),
        ("disconnect",),
    ]


@pytest.mark.asyncio
async def test_proactive_thechat_attachment_only_message_is_delivered():
    result = await _send_to_platform(
        Platform.THECHAT,
        SimpleNamespace(token="token", extra={"base_url": "https://thechat.test"}),
        CONVERSATION_ID,
        "",
        media_files=[("/safe/report.pdf", False)],
    )

    assert result == {"success": True, "message_id": "document-1"}
    assert _FakeTheChatAdapter.instances[0].calls == [
        ("connect",),
        ("document", CONVERSATION_ID, "/safe/report.pdf", None, None),
        ("disconnect",),
    ]


@pytest.mark.asyncio
async def test_proactive_thechat_audio_sends_body_then_attachment():
    result = await _send_to_platform(
        Platform.THECHAT,
        SimpleNamespace(token="token", extra={"base_url": "https://thechat.test"}),
        CONVERSATION_ID,
        "please inspect",
        media_files=[("/safe/sample.wav", False)],
    )

    assert result == {"success": True, "message_id": "voice-1"}
    assert _FakeTheChatAdapter.instances[0].calls == [
        ("connect",),
        ("send", CONVERSATION_ID, "please inspect", None),
        ("voice", CONVERSATION_ID, "/safe/sample.wav", None, None),
        ("disconnect",),
    ]
