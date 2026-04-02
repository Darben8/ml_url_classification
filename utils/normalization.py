import tldextract
def extract_registered_domain(url: str) -> str:
    """
    Extract eTLD+1 from a URL or hostname.
    - https://login.google.com -> google.com
    """
    ext = tldextract.extract(url)

    if not ext.domain or not ext.suffix:
        return None

    return f"{ext.domain}.{ext.suffix}"