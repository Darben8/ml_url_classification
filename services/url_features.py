from urllib.parse import urlparse
import socket
import re
import math

def extract_domain(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        domain = urlparse(url).netloc
        return domain if domain else url.split('/')[0]
    except:
        return url.split('/')[0].split('?')[0]

def has_ip(url):
    return int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)))

def has_at(url):
    return int('@' in url)

def url_length_short(url):
    return int(len(url) < 54)

def count_slash(url):
    return url.count('/')

def has_https(url):
    return int('https' in url.lower())

shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in']
def is_shortened(url):
    return int(any(s in url.lower() for s in shortening_services))

def has_dash_separator(domain):
    return int('-' in domain)

def has_dns(domain):
    try:
        socket.gethostbyname(domain)
        return 1
    except:
        return 0

def count_dots(url):
    return url.count('.')

def count_digits(url):
    return sum(c.isdigit() for c in url)

def count_special_chars(url):
    return sum(not c.isalnum() for c in url)

def extract_tld(domain):
    try:
        parts = domain.split('.')
        return parts[-1] if len(parts) > 1 else ''
    except:
        return ''

def is_tld_com(tld):
    return int(tld == 'com')

def is_tld_org(tld):
    return int(tld == 'org')

def is_tld_net(tld):
    return int(tld == 'net')

def is_tld_edu(tld):
    return int(tld == 'edu')

def is_tld_gov(tld):
    return int(tld == 'gov')

def tld_length(tld):
    return len(tld)

def subdomain_count(domain):
    return len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0

def subdomain_length(domain):
    parts = domain.split('.')
    return len(parts[0]) if len(parts) > 2 else 0

def extract_path(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        return urlparse(url).path
    except:
        return ''

def path_length(url):
    return len(extract_path(url))

def param_count(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        query = urlparse(url).query
        return len(query.split('&')) if query else 0
    except:
        return 0

def has_file_extension(url):
    path = extract_path(url)
    return int(bool(re.search(r'\.(html|php|asp|aspx|jsp|exe|zip)$', path)))

def domain_digit_count(domain):
    return sum(c.isdigit() for c in domain)

suspicious_keywords = ['client', 'verify', 'login', 'secure', 'account', 'update', 'confirm', 'banking', 'signin']
def has_suspicious_keyword(url):
    return int(any(kw in url.lower() for kw in suspicious_keywords))

def domain_entropy(domain):
    import math
    prob = [domain.count(c) / len(domain) for c in set(domain)]
    return -sum([p * math.log2(p) for p in prob if p > 0])

def has_port(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        port = urlparse(url).port
        return int(port is not None)
    except:
        return 0

def url_depth(url):
    path = extract_path(url)
    return path.count('/')


def extract_url_features(url: str) -> dict:
    domain = extract_domain(url)
    tld = extract_tld(domain)

    return {
        # URL-level
        "url_length": len(url),  # full URL length
        "has_ip": has_ip(url),
        "has_https": has_https(url),
        "is_shortened": is_shortened(url),
        "dot_count": count_dots(url),
        "digit_count": count_digits(url),
        "special_char_count": count_special_chars(url),
        "has_suspicious": has_suspicious_keyword(url),
        "has_at": has_at(url),
        "has_file_ext": has_file_extension(url),
        "has_port": has_port(url),
        "param_count": param_count(url),
        "path_length": path_length(url),
        "slash_count": count_slash(url),
        "url_depth": url_depth(url),
        "url_short": url_length_short(url),
        # "url_length": path_length(url),
        # "has_ip": has_ip(url),
        # "has_https": has_https(url),
        # "is_shortened": is_shortened(url),
        # "url_dot_count": count_dots(url),
        # "url_digit_count": count_digits(url),
        # "url_special_char_count": count_special_chars(url),
        # "has_suspicious_keyword": has_suspicious_keyword(url),

        # Domain-level
        #"domain_length": len(domain),
        "dash_in_domain": has_dash_separator(domain),
        "domain_digits": domain_digit_count(domain),
        "entropy": domain_entropy(domain),
        "subdomain_count": subdomain_count(domain),
        "subdomain_length": subdomain_length(domain),
        

        # TLD
        "tld_com": is_tld_com(tld),
        "tld_net": is_tld_net(tld),
        "tld_org": is_tld_org(tld),
        "tld_edu": is_tld_edu(tld),
        "tld_gov": is_tld_gov(tld),
        "tld_length": tld_length(tld),
        # "tld_length": len(tld),
        # "tld_is_com": int(tld == "com"),
        # "tld_is_org": int(tld == "org"),
        # "tld_is_net": int(tld == "net"),
    }