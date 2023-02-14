import requests


_PEXELS_AUTH_KEY = ""


def get_pexels_image(
    query: str,
    amount: int = 1,
    orientation: str = None,
    auth_key: str = _PEXELS_AUTH_KEY,
):
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={amount}"
    headers = {"Authorization": auth_key}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response = response.json()
    if response["total_results"] == 0:
        return "No results found"
    if response["total_results"] < amount:
        return "Not enough results found"
    output = []
    for photo in response["photos"]:
        output.append(
            (
                photo["alt"],
                photo["photographer"],
                photo["url"],
            )
        )
    return output
