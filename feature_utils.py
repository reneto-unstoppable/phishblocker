import pandas as pd
from urllib.parse import urlparse, parse_qs
import tldextract
import re
import math
from collections import Counter # Counter is imported but not used directly in this version. Can be removed if not used elsewhere.
import sys
import traceback

def extract_features(url):
    features = {}

    # Basic URL Parsing - all feature extraction is wrapped in this try-except
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        fragment = parsed_url.fragment

        # Feature 1: URL Length
        features['url_length'] = len(url)

        # Feature 2: Number of dots in URL
        features['num_dots'] = url.count('.')

        # Feature 3: Number of hyphens in URL
        features['num_hyphens'] = url.count('-')

        # Feature 4: Number of subdomains
        try:
            ext = tldextract.extract(url)
            # Filter out empty strings from subdomain parts (e.g., if subdomain is empty)
            subdomain_parts = [part for part in ext.subdomain.split('.') if part]
            features['num_subdomains'] = len(subdomain_parts)
        except Exception:
            # Default to 0 if tldextract fails for some reason
            features['num_subdomains'] = 0

        # Feature 5: TLD Length
        try:
            ext = tldextract.extract(url)
            features['tld_length'] = len(ext.suffix)
        except Exception:
            # Default to 0 if tldextract fails
            features['tld_length'] = 0

        # Feature 6: Presence of IP address in hostname (e.g., http://192.168.1.1/malicious)
        features['has_ip'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0

        # Feature 7: Presence of HTTPS (secure connection)
        features['has_https'] = 1 if parsed_url.scheme == 'https' else 0

        # Feature 8: Presence of '@' symbol (often used to obscure the true domain)
        features['has_at_symbol'] = 1 if '@' in url else 0

        # Feature 9: Number of digits in URL (high number can be suspicious for non-numeric domains)
        features['num_digits'] = sum(c.isdigit() for c in url)

        # Feature 10: Number of special characters (excluding common URL delimiters)
        # Defines characters that are generally part of normal URLs but not alphanumeric
        common_url_delimiters = ['.', '-', '/', ':', '?', '=', '&', '#', '_', '%', '~', '+', '*']
        features['num_special_chars'] = sum(1 for char in url if not char.isalnum() and char not in common_url_delimiters)

        # Feature 11: Path Length (length of the part after the domain)
        features['path_length'] = len(path)

        # Feature 12: Query Length (length of the part after '?' including '?')
        features['query_length'] = len(query)

        # Feature 13: Fragment Length (length of the part after '#' including '#')
        features['fragment_length'] = len(fragment)

        # Feature 14: Is Shortened URL (Revised logic)
        # Checks if the domain itself matches a known shortener or starts with it, making it more precise
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'shorte.st', 'cutt.ly', 'rb.gy', 'adf.ly', 'go.gl']
        is_shortened_calculated = 0
        for s in shorteners:
            if domain == s or parsed_url.netloc.startswith(s + '/'):
                is_shortened_calculated = 1
                break
        features['is_shortened'] = is_shortened_calculated

        # Feature 15: Hostname Length (length of the domain part, e.g., 'openai.com' is 10)
        features['hostname_length'] = len(domain)

        # Feature 16: Number of Parameters in Query String (e.g., ?param1=val1&param2=val2)
        features['num_parameters'] = len(parse_qs(query))

        # Feature 17: Count of 'www' (redundant with num_subdomains in some cases, but can be a standalone indicator)
        features['count_www'] = url.lower().count('www')

        # Feature 18: Count of 'com' - REMOVED/COMMENTED OUT (often leads to false positives for legitimate .com domains)
        # features['count_com'] = url.lower().count('com')

        # Feature 19: Count of 'net' - REMOVED/COMMENTED OUT (similar reason as count_com)
        # features['count_net'] = url.lower().count('net')

        # Feature 20: Count of '@' (redundant with has_at_symbol but kept for consistency if needed by model)
        features['count_at'] = url.count('@')

        # Feature 21: Has Suspicious Words (checks for common phishing keywords in the full URL)
        def has_suspicious_words_func(url_string):
            suspicious_words = [
                'confirm', 'account', 'banking', 'login', 'secure', 'update', 'verify', 'paypal', 'microsoft',
                'amazon', 'apple', 'support', 'alert', 'security', 'webscr', 'cmd', 'dispatch', 'signin',
                'ebay', 'billing', 'invoice', 'payment', 'transfer', 'credit', 'debit', 'card', 'access',
                'claim', 'prize', 'winner', 'rewards', 'free', 'gift', 'password', 'reset', 'recover',
                'urgent', 'suspicious', 'activity', 'unusual', 'fraud', 'hold', 'transaction', 'notification',
                'suspension', 'closure', 'restricted', 'forbidden', 'required'
            ]
            url_lower = url_string.lower()
            for word in suspicious_words:
                if word in url_lower:
                    return 1
            return 0
        features['has_suspicious_words'] = has_suspicious_words_func(url)

        # Feature 22-25: Qty of specific characters in path
        features['qty_slash_path'] = path.count('/')
        features['qty_questionmark_path'] = path.count('?')
        features['qty_ampersand_path'] = path.count('&')
        features['qty_equal_path'] = path.count('=')

        # Feature 26-29: Qty of specific characters in domain (should generally be 0 for clean domains)
        features['qty_slash_domain'] = domain.count('/')
        features['qty_questionmark_domain'] = domain.count('?')
        features['qty_ampersand_domain'] = domain.count('&')
        features['qty_equal_domain'] = domain.count('=')

        # Feature 30-38: Qty of specific characters in full URL
        features['qty_exclamation_url'] = url.count('!')
        features['qty_tilde_url'] = url.count('~')
        features['qty_comma_url'] = url.count(',')
        features['qty_plus_url'] = url.count('+')
        features['qty_asterisk_url'] = url.count('*')
        features['qty_hashtag_url'] = url.count('#')
        features['qty_underscore_url'] = url.count('_')
        features['qty_percent_url'] = url.count('%')
        features['qty_colon_url'] = url.count(':')

        # Feature 39: Number of consecutive digits (e.g., 192.168.1.1 or generated domains)
        features['num_consecutive_digits'] = 1 if re.search(r'\d{2,}', url) else 0

        # Feature 40: Has Port Number (e.g., http://example.com:8080/path)
        features['has_port'] = 1 if re.search(r':\d{2,5}$', parsed_url.netloc) else 0

        # Feature 41: Domain Token Count (number of parts in domain, e.g., 'example.com' has 2)
        features['domain_token_count'] = len(domain.split('.'))

        # Feature 42: Punycode Encoded (internationalized domain names used for homograph attacks)
        features['punycode_encoded'] = 1 if 'xn--' in url.lower() else 0

        # Feature 43: Hostname Entropy (Shannon entropy of the hostname string)
        def calculate_entropy(s):
            if not s:
                return 0
            probabilities = [s.count(c) / len(s) for c in set(s)]
            entropy = -sum(p * math.log2(p) for p in probabilities)
            return entropy
        features['hostname_entropy'] = calculate_entropy(domain)

        # Feature 44: Digit Ratio in Hostname (ratio of digits to total characters in domain)
        features['digit_ratio_hostname'] = sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0

        # Feature 45: Has Suspicious Domain Keywords (checks for phishing keywords within the domain/hostname parts)
        def has_suspicious_domain_keywords_func(url_string):
            domain_suspicious_keywords = [
                'login', 'signin', 'secure', 'account', 'verify', 'update', 'support', 'security', 'webscr',
                'billing', 'payment', 'reset', 'password', 'microsoft', 'paypal', 'amazon', 'apple', 'ebay',
                'alert', 'info', 'confirm', 'service'
            ]
            try:
                parsed = urlparse(url_string)
                hostname_lower = parsed.hostname.lower() if parsed.hostname else ""
                for keyword in domain_suspicious_keywords:
                    if keyword in hostname_lower:
                        return 1

                # Also check parts extracted by tldextract for robustness
                ext = tldextract.extract(url_string)
                domain_parts = [part for part in ext.subdomain.split('.') if part] + [ext.domain] + [ext.suffix]
                for part in domain_parts:
                    part_lower = part.lower()
                    for keyword in domain_suspicious_keywords:
                        if keyword in part_lower:
                            return 1
                return 0
            except Exception:
                # If any parsing fails, assume no suspicious keywords to prevent errors
                return 0

        features['has_suspicious_domain_keywords'] = has_suspicious_domain_keywords_func(url)

        # --- IMPORTANT DEBUG PRINT ---
        print(f"\n--- Extracted Features for {url} ---")
        for key, value in features.items():
            print(f"{key}: {value}")
        print("-------------------------------\n")
        # --- END IMPORTANT DEBUG PRINT ---


        # Ensure features dictionary is not empty (should not happen if parsing is successful)
        if not features:
            print(f"DEBUG: Features dictionary is unexpectedly empty for URL: {url}")
            return pd.DataFrame(), "Empty features dict"

        # Convert the dictionary of features to a Pandas DataFrame for consistent output
        features_df = pd.DataFrame([features])
        return features_df, None # Return DataFrame and no error

    except Exception as e:
        # General error handling for the entire feature extraction process
        print(f"ERROR: Feature extraction failed for URL: {url}. Error: {e}")
        traceback.print_exc(file=sys.stdout) # Print full traceback to console
        return pd.DataFrame(), f"Error: {e}" # Return empty DataFrame and error message
