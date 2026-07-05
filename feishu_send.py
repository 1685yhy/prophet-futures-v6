#!/usr/bin/env python3
"""
Feishu Card sender — Builds and sends Feishu card messages.
Supports: header, div (lark_md), hr, column_set, note, colored text.
"""
import sys, json, os, requests
from datetime import datetime

def _load_env():
    env_file = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

APP_ID = os.getenv("FEISHU_APP_ID", "")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
CHAT_ID = "oc_e9bf3cb98e83f50ad4e71dff71f9dce8"

def get_token():
    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET}, timeout=10
    )
    return resp.json()["tenant_access_token"]

def send_card(title, sections, color="blue", pin=False):
    """Send a Feishu card.
    
    sections: list of dicts, color: blue/red/green/yellow, pin: 置顶消息
    """
    token = get_token()
    
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color
        },
        "elements": []
    }
    
    for sec in sections:
        t = sec.get("type", "div")
        
        if t == "hr":
            card["elements"].append({"tag": "hr"})
        elif t == "note":
            card["elements"].append({
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": sec.get("text", "")}]
            })
        elif t == "header":
            card["elements"].append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**{sec.get('text', '')}**"}
            })
        elif t == "columns":
            cols = []
            for item in sec.get("items", []):
                label = item.get("label", "")
                value = item.get("value", "")
                vcolor = item.get("color", "")
                vprefix = f"<font color='{vcolor}'>" if vcolor else ""
                vsuffix = "</font>" if vcolor else ""
                cols.append({
                    "tag": "column", "width": "weighted", "weight": 1,
                    "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": f"**{label}**\n{vprefix}{value}{vsuffix}"}}]
                })
            card["elements"].append({"tag": "column_set", "flex_mode": "bisect", "background_style": "default", "columns": cols})
        else:
            card["elements"].append({"tag": "div", "text": {"tag": "lark_md", "content": sec.get("text", "")}})
    
    body = {
        "receive_id": CHAT_ID,
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False)
    }
    
    resp = requests.post(
        f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body, timeout=10
    )
    result = resp.json()
    
    # Pin if requested and send succeeded
    if pin and result.get("code") == 0:
        msg_id = result.get("data", {}).get("message_id", "")
        if msg_id:
            requests.post(
                f"https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/pin",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
    
    return result.get("code") == 0, result.get("data", {}).get("message_id", "")


def send_alert(title, text, color="red", pin=True):
    """Send a short alert card, pinned."""
    sections = [
        {"type": "div", "text": text},
        {"type": "note", "text": datetime.now().strftime('%H:%M:%S')}
    ]
    return send_card(title, sections, color=color, pin=pin)


def send_scan(title, positions_text, pnl_text=""):
    """Send a routine scan update (not pinned, blue card)."""
    sections = []
    if positions_text:
        sections.append({"type": "header", "text": "📌 持仓"})
        sections.append({"type": "div", "text": positions_text})
    if pnl_text:
        sections.append({"type": "div", "text": pnl_text})
    return send_card(title, sections, color="blue", pin=False)

def send_report(title, report_text):
    """Parse a markdown report into card sections and send it."""
    from datetime import datetime
    
    sections = []
    lines = report_text.strip().split('\n')
    
    buf = []
    col_items = []
    
    for line in lines:
        s = line.strip()
        
        # Horizontal dividers
        if s.startswith('===') or s.startswith('---') or s.startswith('───') or s.startswith('═══'):
            if buf:
                sections.append({"type": "div", "text": '\n'.join(buf)})
                buf = []
            sections.append({"type": "hr"})
            continue
        
        # Column data (pipe format: label | value)
        if '|' in s and not s.startswith(('#', '-', '*', '`')):
            parts = [p.strip() for p in s.split('|')]
            if 2 <= len(parts) <= 4:
                item = {"label": parts[0]}
                val = parts[1]
                # Check for color markers
                if '🔴' in val or '❌' in val:
                    item["value"] = val
                    item["color"] = "red"
                elif '🟢' in val or '✅' in val:
                    item["value"] = val
                    item["color"] = "green"
                else:
                    item["value"] = val
                col_items.append(item)
                if len(col_items) >= 2:
                    sections.append({"type": "columns", "items": col_items})
                    col_items = []
                continue
        
        buf.append(s)
    
    # Flush remaining
    if col_items:
        sections.append({"type": "columns", "items": col_items})
    if buf:
        sections.append({"type": "div", "text": '\n'.join(buf)})
    
    # Footer
    sections.append({"type": "hr"})
    sections.append({"type": "note", "text": f"Prophet Futures v25 | {datetime.now().strftime('%Y-%m-%d %H:%M')} | 仅供学习参考"})
    
    return send_card(title, sections)

if __name__ == "__main__":
    import sys
    title = sys.argv[1] if len(sys.argv) > 1 else "Prophet Futures"
    text = sys.stdin.read().strip() if not sys.stdin.isatty() else " ".join(sys.argv[2:])
    
    if text:
        ok, msg = send_report(title, text)
        if ok:
            print(f"✅ 已发送: {title}")
        else:
            print(f"❌ 失败: {msg}")
    else:
        # Quick test
        send_card("Prophet Futures 测试", [
            {"type": "div", "text": "📊 **LH 生猪** 做多 6手 @ 11650\n现价 11745 | 浮盈 **+9,120**"},
            {"type": "hr"},
            {"type": "columns", "items": [
                {"label": "LH", "value": "🟢 +9,120", "color": "green"},
                {"label": "JM", "value": "🟢 +216", "color": "green"}
            ]},
            {"type": "note", "text": "测试消息"}
        ])
        print("✅ 测试已发送")
