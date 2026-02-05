# src/features.py
from __future__ import annotations
import math
import re
from urllib.parse import urlparse, unquote
from collections import Counter

COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "info", "biz", "io", "co", "us", "uk", "es", "mx", "gt"
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly",
    "rb.gy", "buff.ly", "s.id", "rebrand.ly"
}

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account",
    "bank", "paypal", "apple", "microsoft", "support", "confirm",
    "password", "billing", "invoice", "webscr", "security"
]

HEX_RE = re.compile(r"%[0-9A-Fa-f]{2}")
IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def _safe_parse(url: str):
    """Parse robusto: intenta arreglar URLs sin esquema."""
    if not isinstance(url, str):
        url = "" if url is None else str(url)

    url = url.strip()
    if not url:
        return urlparse("")

    if "://" not in url:
        url = "http://" + url

    return urlparse(url)


def shannon_entropy(text: str) -> float:
    # Entropía de Shannon en bits
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def relative_entropy_vs_baseline(text: str, baseline: dict[str, float] | None = None) -> float:
    # Entropía relativa (KL divergence) respecto a una distribución base.
    if not text:
        return 0.0

    # baseline simple
    if baseline is None:
        baseline = {}
        for ch in "abcdefghijklmnopqrstuvwxyz":
            baseline[ch] = 0.03
        for ch in "0123456789":
            baseline[ch] = 0.02
        for ch in "-._~:/?#[]@!$&'()*+,;=%":
            baseline[ch] = 0.005

        # normalizar
        s = sum(baseline.values())
        baseline = {k: v / s for k, v in baseline.items()}

    counts = Counter(text.lower())
    n = len(text)

    eps = 1e-12
    kl = 0.0
    for ch, c in counts.items():
        p = c / n
        q = baseline.get(ch, eps)
        kl += p * math.log(p / (q + eps) + eps)
    return float(kl)


def extract_features(url: str) -> dict[str, float]:
    p = _safe_parse(url)
    raw = p.geturl()
    decoded = unquote(raw)

    host = (p.hostname or "").lower()
    path = p.path or ""
    query = p.query or ""
    scheme = (p.scheme or "").lower()

    # componentes
    full = decoded
    host_len = len(host)
    url_len = len(full)
    path_len = len(path)
    query_len = len(query)

    # subdominios
    parts = [x for x in host.split(".") if x]
    num_dots = full.count(".")
    num_subdomains = max(0, len(parts) - 2) if len(parts) >= 2 else 0

    # TLD
    tld = parts[-1] if parts else ""
    tld_is_common = 1.0 if tld in COMMON_TLDS else 0.0

    # sospechas típicas
    has_at = 1.0 if "@" in full else 0.0
    has_ip = 1.0 if (IPV4_RE.match(host) is not None) else 0.0
    has_https = 1.0 if scheme == "https" else 0.0
    has_http = 1.0 if scheme == "http" else 0.0

    num_hyphen = full.count("-")
    num_underscore = full.count("_")
    num_slash = full.count("/")
    num_question = full.count("?")
    num_equal = full.count("=")
    num_amp = full.count("&")
    num_percent = full.count("%")

    # tokens
    digit_count = sum(ch.isdigit() for ch in full)
    alpha_count = sum(ch.isalpha() for ch in full)
    other_count = url_len - digit_count - alpha_count

    digit_ratio = (digit_count / url_len) if url_len else 0.0
    alpha_ratio = (alpha_count / url_len) if url_len else 0.0
    other_ratio = (other_count / url_len) if url_len else 0.0

    # puerto explicito
    has_port = 1.0 if p.port is not None else 0.0

    # acortador
    is_shortener = 1.0 if host in SHORTENERS else 0.0

    # palabras sospechosas
    lower = full.lower()
    suspicious_word_hits = sum(1 for w in SUSPICIOUS_WORDS if w in lower)

    # hex-encoding
    hex_count = len(HEX_RE.findall(full))
    has_hex = 1.0 if hex_count > 0 else 0.0

    # repeticion rara de caracteres
    if url_len:
        counts = Counter(lower)
        max_char_freq = max(counts.values()) / url_len
    else:
        max_char_freq = 0.0

    # entropias
    ent = shannon_entropy(lower)
    rel_ent = relative_entropy_vs_baseline(lower)

    return {
        # Longitudes
        "url_len": float(url_len),
        "host_len": float(host_len),
        "path_len": float(path_len),
        "query_len": float(query_len),

        # Conteos simples
        "num_dots": float(num_dots),
        "num_subdomains": float(num_subdomains),
        "num_hyphen": float(num_hyphen),
        "num_underscore": float(num_underscore),
        "num_slash": float(num_slash),
        "num_question": float(num_question),
        "num_equal": float(num_equal),
        "num_amp": float(num_amp),
        "num_percent": float(num_percent),

        # Composición
        "digit_ratio": float(digit_ratio),
        "alpha_ratio": float(alpha_ratio),
        "other_ratio": float(other_ratio),
        "max_char_freq": float(max_char_freq),

        # Flags
        "has_at": float(has_at),
        "has_ip": float(has_ip),
        "has_https": float(has_https),
        "has_http": float(has_http),
        "has_port": float(has_port),
        "tld_is_common": float(tld_is_common),
        "is_shortener": float(is_shortener),
        "has_hex": float(has_hex),

        # Heuristicas de contenido
        "suspicious_word_hits": float(suspicious_word_hits),
        "hex_count": float(hex_count),

        # Entropias
        "shannon_entropy": float(ent),
        "relative_entropy": float(rel_ent),
    }
