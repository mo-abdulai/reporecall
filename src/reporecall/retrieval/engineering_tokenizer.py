import re
from urllib.parse import urlsplit

_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_LEXEME = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")
_CAMEL_PART = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)
_IDENTIFIER_SEPARATOR = re.compile(r"[._/-]+")


class EngineeringTokenizer:
    """Tokenize prose while retaining exact software-engineering identifiers."""

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Return deterministic case-insensitive tokens without stemming."""

        url_reduced = _URL.sub(_url_fragments, text.replace("\\", "/"))
        tokens: list[str] = []
        for match in _LEXEME.finditer(url_reduced):
            lexeme = match.group(0).strip("._/-")
            if not lexeme:
                continue
            tokens.extend(_lexeme_tokens(lexeme))
        return tuple(tokens)


def _url_fragments(match: re.Match[str]) -> str:
    raw_url = match.group(0).rstrip(".,;:!?)]}")
    parsed = urlsplit(raw_url)
    hostname = parsed.hostname or ""
    path_parts = " ".join(part for part in parsed.path.split("/") if part)
    return f" {hostname} {path_parts} "


def _lexeme_tokens(lexeme: str) -> tuple[str, ...]:
    candidates = [lexeme]
    candidates.extend(
        component
        for component in lexeme.split("/")
        if component and component != lexeme
    )

    for component in tuple(candidates[1:] or candidates):
        candidates.extend(
            part
            for part in _IDENTIFIER_SEPARATOR.split(component)
            if part and part != component
        )

    for component in tuple(candidates):
        candidates.extend(
            part
            for part in _CAMEL_PART.findall(component)
            if part and part != component
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        token = candidate.casefold()
        if token and token not in seen:
            seen.add(token)
            normalized.append(token)
    return tuple(normalized)
