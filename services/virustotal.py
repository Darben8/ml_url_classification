import os
import requests
import base64
from services.vtrate_limit import rate_limit
from services.vtcache import get_cached_vt, save_vt_cache, init_vt_cache
from dotenv import load_dotenv
load_dotenv()
init_vt_cache()

vt_api_key = os.getenv("VIRUSTOTAL_API_KEY")
base_url = "https://www.virustotal.com/api/v3"
headers = {
    "X-Apikey":vt_api_key
}

def vt_check_url(url: str) -> dict:
    # Check cache first
    cached_response = get_cached_vt(url)
    if cached_response:
        return cached_response
    
    # Rate limit API calls
    rate_limit()
    url_id = base64.urlsafe_b64encode(
        url.encode()
    ).decode().rstrip("=")

    response = requests.get(
        f"{base_url}/urls/{url_id}",
        headers=headers,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()["data"]["attributes"]
    stats = data.get("last_analysis_stats", {})

    malicious = int(stats.get("malicious", 0))
    suspicious = int(stats.get("suspicious", 0))
    harmless = int(stats.get("harmless", 0))
    undetected = int(stats.get("undetected", 0))
    total = malicious + suspicious + harmless + undetected

    # Detection rate (bounded, numeric feature)
    vt_detection_rate = (
        (malicious + (0.5 * suspicious)) / total #0.5 because suspicious is less severe than malicious
        if total > 0 else 0.0
    )
    
    parsed_result = {
        #"vt_analysis_stats": stats,
        "vt_malicious_count": malicious,
        "vt_suspicious_count": suspicious,
        "vt_harmless_count": harmless,
        "vt_undetected_count": undetected,
        "vt_total_engines": total,
        "vt_detection_rate": round(vt_detection_rate, 4),
    }

    save_vt_cache(url, parsed_result)
    return parsed_result

