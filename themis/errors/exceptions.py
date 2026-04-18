class JudgeParseError(Exception):
    """Raised when the LLM judge response yields no parseable judgments."""

    def __init__(self, provider: str, candidate_count: int, raw_response: str):
        self.provider = provider
        self.candidate_count = candidate_count
        self.raw_response = raw_response
        super().__init__(
            f"Judge returned no parseable judgments for {candidate_count} candidates "
            f"using provider '{provider}'."
        )


class EmptyPetitionError(Exception):
    """Raised when a PDF yields no extractable text."""

    def __init__(self) -> None:
        super().__init__("The uploaded PDF contains no extractable text.")
