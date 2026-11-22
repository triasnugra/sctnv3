"""Notifikasi opsional (Telegram / Discord) untuk signal BARU. Aktif hanya jika secret/env diisi.
Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DISCORD_WEBHOOK_URL"""
from __future__ import annotations

import logging
import os
from typing import List

import requests

from .models import Signal
from .utils import fmt_price

log = logging.getLogger("engine.notify")


def format_signal(s: Signal) -> str:
    p = s.plan
    lines = [f"{'🟢' if s.action == 'LONG' else '🔴'} {s.symbol} {s.action} ({s.status}) — skor {s.score:g}/{s.max_score:g}",
             f"Trigger: {s.primary.kind} [{s.primary.tf}] · Bias HTF: {s.bias}"]
    if p:
        lines += [f"Entry {fmt_price(p.entry)} | Stop {fmt_price(p.stop)} ({p.stop_pct:.2f}%)",
                  f"TP1 {fmt_price(p.tp1)} ({p.rr1:.1f}R) | TP2 {fmt_price(p.tp2)} ({p.rr2:.1f}R) | RR {p.blended_rr:.2f}",
                  f"Lev maks aman {p.leverage}x"]
    lines.append("Analisis otomatis berbasis hipotesis — bukan nasihat finansial.")
    return "\n".join(lines)


def send_alerts(signals: List[Signal]) -> int:
    if not signals:
        return 0
    tok, chat, hook = (os.getenv(k, "").strip() for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DISCORD_WEBHOOK_URL"))
    if not ((tok and chat) or hook):
        log.info("Notifikasi dilewati (secret belum diisi).")
        return 0
    sent = 0
    for s in signals:
        text = format_signal(s)
        try:
            if tok and chat:
                requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": text}, timeout=15).raise_for_status()
                sent += 1
            if hook:
                requests.post(hook, json={"content": text}, timeout=15).raise_for_status()
                sent += 1
        except Exception as exc:  # noqa: BLE001 - notifikasi tidak boleh mematikan run
            log.warning("Notifikasi gagal untuk %s: %s", s.symbol, exc)
    return sent
