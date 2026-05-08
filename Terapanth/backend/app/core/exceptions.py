# app/core/exceptions.py
# Domain exception classes for the Terapanth platform.
# Imported by: all service modules. Handled globally in app/main.py.


class TerapanthError(Exception):
    """Base exception for all domain errors."""


class DocumentNotFoundError(TerapanthError):
    def __init__(self, document_id: str):
        super().__init__(f"Document {document_id} not found")
        self.document_id = document_id


class DocumentTooLargeError(TerapanthError):
    def __init__(self, size_bytes: int, max_bytes: int):
        super().__init__(f"File {size_bytes} bytes exceeds limit {max_bytes} bytes")


class DocumentProcessingError(TerapanthError):
    """Raised when chunking or embedding fails."""


class ConversationNotFoundError(TerapanthError):
    def __init__(self, conversation_id: str):
        super().__init__(f"Conversation {conversation_id} not found")
        self.conversation_id = conversation_id


class SessionNotFoundError(TerapanthError):
    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} not found")
        self.session_id = session_id


class RetrievalError(TerapanthError):
    """Raised when retrieval or reranking fails."""


class LLMError(TerapanthError):
    """Raised when LLM call fails."""
