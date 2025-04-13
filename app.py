import sqlite3
import hashlib
import requests
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from steam_api import get_owned_games
import uvicorn
from src.infer_final_ver import load_resources, recommend_for_user  
from pathlib import Path
import json


app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_steam_game_name(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    response = requests.get(url)
    data = response.json()
    
    if str(app_id) in data and data[str(app_id)]['success']:
        return data[str(app_id)]['data'].get('name', 'Unknown Game')
    return "Unknown Game"

def init_db():
    with sqlite3.connect("user_info.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            password TEXT DEFAULT '000000'
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS histories (
            user_id INTEGER,
            date TEXT,
            app_id INTEGER,
            is_recommended BOOLEAN,
            cumulative_review INTEGER,
            cumulative_recommended INTEGER,
            cumulative_not_recommend INTEGER,
            recommend_ratio REAL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            user_id INTEGER,
            app_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
        conn.commit()

def get_db_connection():
    conn = sqlite3.connect("user_info.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
def login(request: Request, user_id: int = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id=? AND password=?", (user_id, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return RedirectResponse(url=f"/dashboard?user={user_id}", status_code=302)
    else:
        return HTMLResponse("<h3>Invalid user ID/password. <a href='/'>Try Again</a></h3>")
    
# ---------------------------
# Register Route
# ---------------------------


# 加载 appid_dic.json
with open("small_10/appid_dic.json", "r") as f:
    appid_dic = json.load(f)
    appid_dic = {int(k): v for k, v in appid_dic.items()}

# 加载 appid_user_dic.json
appid_user_dic_path = Path("small_10/appid_user_dic.json")
if appid_user_dic_path.exists():
    with open(appid_user_dic_path, "r") as f:
        appid_user_dic = json.load(f)
        appid_user_dic = {int(k): v for k, v in appid_user_dic.items()}
else:
    appid_user_dic = {}

@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(request: Request, user_id: int = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Passwords do not match."})

    with sqlite3.connect("user_info.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        if cursor.fetchone():
            return templates.TemplateResponse("register.html", {"request": request, "error": "User already exists."})

        cursor.execute("INSERT INTO users (user_id, password) VALUES (?, ?)", (user_id, password))

        try:
            games = get_owned_games(str(user_id))
            today = datetime.now().strftime('%Y-%m-%d')

            user_appids = []

            for game in games.get('response', {}).get('games', []):
                app_id = game.get('appid')
                user_appids.append(app_id)

                if app_id in appid_dic:
                    review_data = appid_dic[app_id]
                    cursor.execute("""
                        INSERT INTO histories (
                            user_id, date, app_id, is_recommended, 
                            cumulative_review, cumulative_recommended, 
                            cumulative_not_recommend, recommend_ratio
                        ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?)
                    """, (
                        user_id, today, app_id,
                        review_data[0],
                        review_data[1],
                        review_data[2],
                        review_data[3]
                    ))
                else:
                    cursor.execute("""
                        INSERT INTO histories (
                            user_id, date, app_id, is_recommended, 
                            cumulative_review, cumulative_recommended, 
                            cumulative_not_recommend, recommend_ratio
                        ) VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                    """, (user_id, today, app_id))

            # 更新 appid_user_dic
            if user_id not in appid_user_dic:
                appid_user_dic[user_id] = user_appids

                #  写入时 key 转为 str，保持 JSON 合规
                with open(appid_user_dic_path, "w") as f:
                    json.dump({str(k): v for k, v in appid_user_dic.items()}, f)

        except Exception as e:
            print(" Steam API error:", e)

        conn.commit()

    return RedirectResponse(url=f"/dashboard?user={user_id}", status_code=303)




@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, user: int):
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.get("/history", response_class=HTMLResponse)
def history(request: Request, user: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, app_id, is_recommended, cumulative_review, cumulative_recommended, 
               cumulative_not_recommend, recommend_ratio 
        FROM histories WHERE user_id = ? ORDER BY date DESC
    """, (user,))
    history_records = cursor.fetchall()
    conn.close()

    # 获取游戏名称
    history_with_names = []
    for entry in history_records:
        game_name = get_steam_game_name(entry[1])
        history_with_names.append((*entry, game_name))

    return templates.TemplateResponse(
        "history.html", 
        {"request": request, "user": user, "history": history_with_names}
    )

resources = load_resources()
@app.get("/recommend", response_class=HTMLResponse)
def recommend(request: Request, user: int):
    try:
        recommendations = recommend_for_user(user, resources, topk=10)
    except ValueError as e:
        return HTMLResponse(f"<h3>{str(e)} <a href='/dashboard?user={user}'>Back</a></h3>")

    # 获取游戏名称
    recommendations_with_names = []
    for appid, score in recommendations:
        game_name = get_steam_game_name(appid)
        recommendations_with_names.append((appid, game_name, round(score, 4)))  # 可以显示得分

    return templates.TemplateResponse(
        "recommend.html", 
        {"request": request, "user": user, "recommendations": recommendations_with_names}
    )



if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
