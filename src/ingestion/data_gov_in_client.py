import os
from typing import Any, Dict, List

import requests


BASE_URL = "https://api.data.gov.in/resource"


class DataGovInClient:
    def __init__(self, api_key: str, timeout_seconds: int = 30) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def fetch_records(
        self,
        resource_id: str,
        offset: int = 0,
        limit: int = 100,
        data_format: str = "json",
    ) -> List[Dict[str, Any]]:
        url = f"{BASE_URL}/{resource_id}"
        params = {
            "api-key": self.api_key,
            "format": data_format,
            "offset": offset,
            "limit": limit,
        }

        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()

        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError("Unexpected response shape: 'records' is not a list")
        return records


def get_client_from_env() -> DataGovInClient:
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        raise ValueError("DATA_GOV_IN_API_KEY is missing in environment")
    return DataGovInClient(api_key=api_key)
