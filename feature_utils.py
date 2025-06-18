import pandas as pd

def extract_features(url):
    url_length = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(not c.isalnum() for c in url)
    num_subdomains = url.count('.') - 1
    count_at = url.count('@')
    count_com = url.count('.com')
    count_hyphen = url.count('-')
    count_net = url.count('.net')
    count_www = url.count('www')
    has_https = int(url.startswith("https"))

    features = {
        "url_length": url_length,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "num_subdomains": num_subdomains,
        "count_at": count_at,
        "count_com": count_com,
        "count_hyphen": count_hyphen,
        "count_net": count_net,
        "count_www": count_www,
        "has_https": has_https
    }

    return pd.DataFrame([features])
