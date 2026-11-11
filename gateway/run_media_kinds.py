"""Audio attachment kinds: native voice notes (STT) versus ordinary audio files."""
from gateway.config import Platform
from gateway.platforms.event import MessageType


def audio_mime_is_stt_input(event, mime: str) -> bool:
    """STT verdict for an attachment whose message-level type did not already decide it."""
    # TheChat has no native voice-note attachment kind. Its message-level type
    # reflects the highest-priority item in a mixed batch, so an audio file sent
    # beside an image arrives as PHOTO and must retain file semantics.
    source = getattr(event, "source", None)
    if getattr(source, "platform", None) == Platform.THECHAT:
        return False

    # Preserve latest-main behavior for platforms whose native voice notes are
    # identified by an audio MIME even when their message-level type is TEXT.
    return mime.startswith("audio/")


def _event_media_is_voice(event, index: int) -> bool:
    """True only for platform-native voice notes that should go through STT."""
    from gateway.run import _event_media_is_audio
    return (
        _event_media_is_audio(event, index)
        and getattr(event, "message_type", None) == MessageType.VOICE
    )


def _event_media_is_audio_file(event, index: int) -> bool:
    """True for ordinary audio files, including mixed TheChat batches."""
    from gateway.run import _event_media_is_audio, _event_media_is_stt_input
    return _event_media_is_audio(event, index) and not _event_media_is_stt_input(
        event, index
    )
