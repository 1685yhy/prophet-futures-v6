#!/usr/bin/env python3
"""飞书卡片消息发送工具 — 共享模块"""
import json, os, urllib.request, urllib.error

ENV_FILE = os.path.expanduser("~/.hermes/.env")

def _load_env():
    """从 Hermes .env 读取飞书凭证"""
    env = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

def _get_token():
    """获取 tenant_access_token"""
    env = _load_env()
    app_id = env.get("FEISHU_APP_ID", "")
    app_secret = env.get("FEISHU_APP_SECRET", "")
    if not app_id or not app_secret:
        return None

    req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        return data.get("tenant_access_token")

_open_id_cache = None

def _get_open_id():
    """获取用于接收消息的 open_id（ou_ 开头）"""
    global _open_id_cache
    if _open_id_cache:
        return _open_id_cache

    env = _load_env()
    allowed = env.get("FEISHU_ALLOWED_USERS", "")

    # 如果已经是 open_id 格式，直接返回
    if allowed.startswith("ou_"):
        _open_id_cache = allowed
        return allowed

    # 用 user_id 先获取 token，再调 API 拿 open_id
    app_id = env.get("FEISHU_APP_ID", "")
    app_secret = env.get("FEISHU_APP_SECRET", "")
    if not app_id or not app_secret:
        return allowed

    try:
        # 获取新 token
        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            token = json.loads(resp.read()).get("tenant_access_token", "")

        if not token:
            return allowed

        # 查用户 open_id
        req2 = urllib.request.Request(
            f"https://open.feishu.cn/open-apis/contact/v3/users/{allowed}?user_id_type=user_id",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req2) as resp2:
            data = json.loads(resp2.read())
            open_id = data.get("data", {}).get("user", {}).get("open_id", "")
            if open_id:
                _open_id_cache = open_id
                return open_id
    except Exception as e:
        pass

    return allowed

def send_card(title, elements, template="blue", subtitle=None):
    """发送飞书卡片消息（旧版格式，兼容性好）
    
    Args:
        title: 卡片标题
        elements: 卡片元素列表，每项 {"tag": "div", "text": {"tag": "lark_md", "content": "..."}}
        template: 标题颜色 (blue/red/green/purple)
        subtitle: 副标题（可选）
    
    Returns:
        (success: bool, message: str)
    """
    token = _get_token()
    if not token:
        return False, "无法获取飞书token，检查.env中的FEISHU_APP_ID/FEISHU_APP_SECRET"

    open_id = _get_open_id()
    if not open_id:
        return False, "找不到接收人，检查FEISHU_ALLOWED_USERS"

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template
        },
        "elements": elements
    }
    if subtitle:
        card["header"]["subtitle"] = {"tag": "plain_text", "content": subtitle}

    payload = {
        "receive_id": open_id,
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False)
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            if result.get("code") == 0:
                return True, result["data"]["message_id"]
            return False, f"API错误({result.get('code')}): {result.get('msg', 'unknown')}"
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()[:300]
        return False, f"HTTP {e.code}: {err_body}"
    except Exception as e:
        return False, str(e)

def md(text):
    """快捷创建 lark_md div 元素"""
    return {"tag": "div", "text": {"tag": "lark_md", "content": text}}

def hr():
    """分隔线"""
    return {"tag": "hr"}

def section(title, body):
    """创建一个带标题的区块"""
    elems = [md(f"**{title}**\n\n{body}")]
    return elems

# ── 下面是根据计算逻辑生成具体的操作建议文本 ──

def build_position_actions(positions, market_data, sym_configs, mode="morning"):
    """生成持仓操作建议 — 每品种一句话结论 + 一个明确操作
    
    格式：
      🟢 LH 做多 3手 | +620点 (2.7 ATR) | 距止损45点 ⚠️ 
      → 减1手至2手，止损上移至12300
    """
    lines = []
    for sym_key, m in market_data.items():
        cfg = sym_configs.get(sym_key, {})
        pos = positions.get(sym_key)
        name = cfg.get('name', sym_key)
        atr = m['atr']
        price = m['price']
        trend = m.get('trend', '↔️')

        if pos:
            d = pos['dir']
            entry = pos['entry']
            vol = pos['vol']
            stop = pos['stop']
            tp = pos['take_profit']

            if d == 'LONG':
                pnl_pts = price - entry
                dist_stop = price - stop
            else:
                pnl_pts = entry - price
                dist_stop = stop - price

            pnl_atr = pnl_pts / atr if atr > 0 else 0
            dist_atr = dist_stop / atr if atr > 0 else 0
            emoji = "🟢" if pnl_pts > 0 else "🔴"
            dir_cn = "做多" if d == 'LONG' else "做空"

            # 风险紧密度
            if dist_atr < 0.5:
                risk_tag = f"距止损{dist_stop:.0f}点 🚨"
            elif dist_atr < 1.0:
                risk_tag = f"距止损{dist_stop:.0f}点 ⚠️"
            elif dist_atr < 2.0:
                risk_tag = f"距止损{dist_stop:.0f}点"
            else:
                risk_tag = f"安全垫{dist_stop:.0f}点"

            # 一行摘要
            lines.append(f"{emoji} **{name}** {dir_cn} {vol}手 | {'+' if pnl_pts>0 else ''}{pnl_pts:.0f}点 ({pnl_atr:.1f}ATR) | {risk_tag}")
            lines.append(f"　成本{entry:.0f} → 现价{price:.0f} | 止损{stop:.0f} | 止盈{tp:.0f}")

            # ── 一个明确操作 ──
            be_atr = cfg.get('be_atr', 1.0)
            reduce1_atr = cfg.get('reduce1_atr', 2.0)
            reduce2_atr = cfg.get('reduce2_atr', 4.0)

            if dist_atr < 0.5:
                # 🚨 止损即将触发
                lines.append(f"→ 🚨 **止损在即**: 距止损仅 {dist_stop:.0f} 点，一旦触发立即出场。不要加仓。")
            elif pnl_atr < be_atr:
                target = entry + atr * be_atr if d == 'LONG' else entry - atr * be_atr
                remain = target - price if d == 'LONG' else price - target
                lines.append(f"→ **持有**: 保本触发价 {target:.0f} (还需 {remain:.0f} 点)")
            elif pnl_atr < reduce1_atr:
                remain = reduce1_atr * atr - pnl_pts
                target = price + remain if d == 'LONG' else price - remain
                lines.append(f"→ **持有**: 减仓触发 {target:.0f} (还需 {remain:.0f} 点)，止损已保本")
            elif pnl_atr < reduce2_atr:
                cut = max(1, int(vol * cfg.get('reduce1_pct', 0.5)))
                keep = vol - cut
                remain = reduce2_atr * atr - pnl_pts
                target = price + remain if d == 'LONG' else price - remain
                lines.append(f"→ ⚠️ **减仓**: 建议减 {cut}手 → 留 **{keep}手**。下档 {target:.0f} 再减")
            else:
                keep = max(1, int(vol * (1 - cfg.get('reduce2_pct', 0.5))))
                lines.append(f"→ 🔔 **锁利**: 大幅盈利，建议减至 {keep}手 锁利润")

        else:
            atr_pct = m.get('atr_pct', atr / price)
            if atr_pct < 0.01: lev = 3.0
            elif atr_pct < 0.02: lev = 2.0
            elif atr_pct < 0.03: lev = 1.5
            elif atr_pct < 0.05: lev = 0.5
            else: lev = 0
            vol = max(1, int(lev * (cfg['max_pos'] // 2))) if lev > 0 else 0

            if vol > 0:
                sm = cfg['stop_mult']
                rr = cfg['rr']
                ls = price - atr * sm
                ss = price + atr * sm
                lt = price + (price - ls) * rr
                st = price - (ss - price) * rr
                mg = vol * price * cfg['multiplier'] * 0.15
                lines.append(f"⚪ **{name}** 空仓 | 现价 {price:.0f} | {trend}")
                lines.append(f"→ 🟢 做多: 入场{price:.0f} 止损**{ls:.0f}** 止盈**{lt:.0f}** | {vol}手 ¥{mg/10000:.1f}万")
                lines.append(f"→ 🔴 做空: 入场{price:.0f} 止损**{ss:.0f}** 止盈**{st:.0f}** | {vol}手 ¥{mg/10000:.1f}万")
            else:
                lines.append(f"⚪ **{name}** 空仓 | 观望 (波动 {atr_pct:.1%} 过大)")

        lines.append("")

    return "\n".join(lines).rstrip()

def build_market_table(market_data, sym_configs):
    """生成行情表格"""
    rows = ["| 品种 | 现价 | 涨跌 | ATR | 趋势 |",
            "|------|------|------|-----|------|"]
    for sym_key, m in market_data.items():
        name = sym_configs.get(sym_key, {}).get('name', sym_key)
        rows.append(
            f"| {name} | {m['price']:.0f} | {m['chg']:+.1%} | {m['atr']:.0f} | {m.get('trend','↔️')} |"
        )
    return "\n".join(rows)
