import requests

API_KEY = '718B03D709901F179EDD3C94F082D575'
BASE_URL = 'https://api.steampowered.com'

def get_owned_games(steam_id):
    url = f"{BASE_URL}/IPlayerService/GetOwnedGames/v1/"
    params = {
        'key': API_KEY,
        'steamid': steam_id,
        'include_appinfo': 1,
        'include_played_free_games': 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()  # 通常包含 response.games
    except requests.RequestException as e:
        print("Error fetching Steam data:", e)
        return {}