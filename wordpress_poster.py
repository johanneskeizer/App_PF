import requests
import yaml

def load_wp_config(path="wp_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def post_to_wordpress(content, title=None, config_path="wp_config.yaml"):
    config = load_wp_config(config_path)
    base_url = config["WP_BASE_URL"].rstrip("/")
    username = config["WP_USERNAME"]
    password = config["WP_APP_PASSWORD"]
    status = config.get("POST_STATUS", "draft")
    title = title or config.get("DEFAULT_TITLE", "Post from CPA")

    url = f"{base_url}/wp-json/wp/v2/posts"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "title": title,
        "content": content,
        "status": status
    }

    try:
        response = requests.post(url, json=payload, auth=(username, password), headers=headers)
        response.raise_for_status()
        result = response.json()
        return {"success": True, "id": result.get("id"), "link": result.get("link")}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}
