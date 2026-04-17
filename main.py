#!/usr/bin/env python3
"""
Fraud detection pipeline v3 — LLM-centric with full user context + audio transcription.

Architecture:
  1. Load all data (CSV, JSON, discover audio files)
  2. Transcribe audio (whisper, cached to JSON)
  3. Build per-user dossiers (profile + comms timeline + all transactions)
  4. For each user: enrich with deterministic features → LLM fraud analysis
  5. Output: fraud_transactions.txt

Key improvements over v2:
  - Fixed BIOTAG detection (now handles Unicode: Ö, É, etc.) — 479 txs were missing GPS
  - Audio transcription reveals scam/extortion calls with exact timestamps
  - LLM sees ALL transactions per user (not pre-filtered by ML)
  - Complete communication timeline with temporal phishing→transaction correlation
  - Per-user analysis with full behavioral context
  - User descriptions (phishing vulnerability, travel patterns) passed to LLM

Usage:
    python main.py "Brave New World - train" "Deus Ex - train" "The Truman Show - train"
"""

import sys
import os
import json
import re
import glob
import unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict
from bisect import bisect_left
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
import ulid
from tqdm import tqdm

try:
    import whisper as whisper_lib
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

load_dotenv()


# ═══════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════

def is_biotag(s):
    """Check if string matches biotag pattern: XXX-XXX-XXX-XXX-N (Unicode-safe)."""
    parts = str(s).split('-')
    return len(parts) == 5 and len(parts[4]) == 1 and parts[4].isdigit()


def haversine_km(lat1, lng1, lat2, lng2):
    try:
        lat1, lng1, lat2, lng2 = map(float, [lat1, lng1, lat2, lng2])
    except (ValueError, TypeError):
        return np.nan
    R = 6371
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


PHISHING_KEYWORDS = [
    'urgent', 'verify', 'suspend', 'locked', 'click here',
    'update payment', 'unusual activity', 'confirm identity',
    'avoid suspension', 'secure your', 'verify now', 'unauthorized',
    'account has been', 'immediately', 'security alert',
    'action required', 'limited time', 'expire', 'deactivat',
    'paypa1', 'amaz0n', 'paypai', 'amazom',
    'billing department', 'recover your', 'compromis', 'footage',
    'won a prize', 'lottery', 'inheritance', 'beneficiary',
    'confirm your', 'verify your', 'legal action',
    'blackmail', 'threaten', 'hack', 'breach', 'freeze',
    'escrow', 'safe account', 'fraud unit', 'fraud',
    'bank security', 'suspicious', 'frozen',
]


def classify_comm(text):
    """Classify communication as PHISHING or NORMAL."""
    lower = text.lower()
    return 'PHISHING' if any(w in lower for w in PHISHING_KEYWORDS) else 'NORMAL'


def safe_ts(val):
    """Convert any timestamp to tz-naive pd.Timestamp for safe comparison."""
    try:
        t = pd.Timestamp(val)
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        return t
    except Exception:
        return None


SYSTEM_PROMPT = """You are a forensic fraud analyst investigating banking transactions for a fraud detection system.

For each user you receive their COMPLETE dossier: profile, ALL communications, and ALL transactions.

YOUR TASK: Identify transactions that are TRUE FRAUD — return ONLY their transaction_ids as a JSON array.

═══ FRAUD INDICATORS (strongest to weakest) ═══

1. PHISHING → TRANSACTION CHAIN (strongest signal):
   User received phishing SMS, scam email, or extortion phone call → then made an unusual
   transfer/payment within days matching the scam's demands. The user's phishing vulnerability
   from their profile tells you how likely they fell for it.

2. SOCIAL ENGINEERING CALL → PAYMENT:
   Audio transcript shows extortion/blackmail demand for specific amount → check if a matching
   payment left the account within days.

3. LOCATION + AMOUNT ANOMALY:
   Transaction occurs far from home (>100km) AND amount is unusual for this user.
   Cross-reference: does the user travel for work? (consultants, freelancers do)

4. UNKNOWN RECIPIENT + UNUSUAL AMOUNT:
   Transfer to an external account (not a known user) with no clear purpose,
   especially after receiving phishing communications.

5. PATTERN BREAK:
   Sudden change in spending behavior — e.g., normally spends €50-200, then sends €3000.

═══ NOT FRAUD (common false positives — do NOT flag) ═══

- Salary payments (from EMP codes), rent, insurance, subscriptions, utility bills
- ATM withdrawals at typical amounts (even at night or unusual locations if traveling)
- Store/restaurant purchases near home at normal amounts
- Automated direct debits at any hour
- Travel spending for users who travel for work
- Purchases that are slightly above average but have clear legitimate descriptions
- Utility bill payments (even if mentioned in a scam call — the real utility payment is legit)

═══ KEY REASONING ═══

- A transaction being statistically unusual is NOT enough — it needs a TRIGGERING EVENT (phishing/scam)
- Users with HIGH phishing vulnerability (>40%) who received scam comms ARE likely victims
- Look at the TEMPORAL CHAIN: scam communication date → suspicious transaction within 1-14 days
- The scam often demands a SPECIFIC AMOUNT — check if a transfer near that amount happened after
- Multiple fraud transactions can happen to the same user (repeated victimization)

Respond with ONLY a JSON array of transaction_id strings. No explanation.
Example: ["tx-id-1", "tx-id-2"]
If no fraud: []"""


# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_data(folder):
    df = pd.read_csv(os.path.join(folder, 'transactions.csv'))
    with open(os.path.join(folder, 'users.json')) as f:
        users = json.load(f)
    with open(os.path.join(folder, 'locations.json')) as f:
        locations = json.load(f)
    with open(os.path.join(folder, 'mails.json')) as f:
        mails = json.load(f)
    with open(os.path.join(folder, 'sms.json')) as f:
        sms = json.load(f)

    audio_dir = os.path.join(folder, 'audio')
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.mp3'))) if os.path.isdir(audio_dir) else []

    return df, users, locations, mails, sms, audio_files


# ═══════════════════════════════════════════════════════════════
# STEP 2 — TRANSCRIBE AUDIO (with cache)
# ═══════════════════════════════════════════════════════════════

def transcribe_audio(audio_files, folder):
    """Transcribe mp3 files using whisper. Results cached to audio_transcripts.json."""
    if not audio_files:
        return {}

    cache_path = os.path.join(folder, 'audio_transcripts.json')

    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        cached_files = {t['file'] for ts in cached.values() for t in ts}
        needed_files = {os.path.basename(f) for f in audio_files}
        if needed_files <= cached_files:
            print(f"      Using cached transcripts ({len(needed_files)} files)")
            return cached

    if not WHISPER_AVAILABLE:
        print("      ⚠ whisper not available, skipping audio transcription")
        return {}

    print(f"      Loading whisper model...")
    model = whisper_lib.load_model("base")

    transcripts = defaultdict(list)
    for fpath in tqdm(audio_files, desc="      Transcribing", unit="file"):
        fname = os.path.basename(fpath)
        match = re.match(r'(\d{8})_(\d{6})-(.+)\.mp3', fname)
        if not match:
            continue
        date_str, time_str, username = match.group(1), match.group(2), match.group(3)
        timestamp = (f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T"
                     f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}")

        result = model.transcribe(fpath)
        transcripts[username].append({
            'timestamp': timestamp,
            'transcript': result['text'].strip(),
            'file': fname,
        })

    for user in transcripts:
        transcripts[user].sort(key=lambda x: x['timestamp'])

    # Cache results
    with open(cache_path, 'w') as f:
        json.dump(dict(transcripts), f, indent=2, ensure_ascii=False)
    print(f"      Cached transcripts to {cache_path}")

    return dict(transcripts)


# ═══════════════════════════════════════════════════════════════
# STEP 3 — MAP COMMUNICATIONS TO USERS
# ═══════════════════════════════════════════════════════════════

def map_sms_to_users(sms_raw, users):
    """Map SMS to user IBANs using name + city matching."""
    name_to_iban = {}
    city_to_iban = {}
    for u in users:
        name_to_iban[u['first_name'].lower()] = u['iban']
        name_to_iban[f"{u['first_name']} {u['last_name']}".lower()] = u['iban']
        city_to_iban[u['residence']['city'].lower()] = u['iban']

    phone_to_iban = {}
    for s in sms_raw:
        text = s['sms']
        to_m = re.search(r'To: (.+)', text)
        if not to_m:
            continue
        phone = to_m.group(1).strip()
        if phone in phone_to_iban:
            continue
        for name, iban in name_to_iban.items():
            if name in text.lower():
                phone_to_iban[phone] = iban
                break
        if phone not in phone_to_iban:
            for city, iban in city_to_iban.items():
                if city in text.lower():
                    phone_to_iban[phone] = iban
                    break

    sms_by_iban = defaultdict(list)
    for s in sms_raw:
        text = s['sms']
        to_m = re.search(r'To: (.+)', text)
        if to_m:
            phone = to_m.group(1).strip()
            iban = phone_to_iban.get(phone)
            if iban:
                frm = re.search(r'From: (.+)', text)
                date = re.search(r'Date: ([\d-]+ [\d:]+)', text)
                msg = re.search(r'Message: (.+?)$', text, re.DOTALL)
                sms_by_iban[iban].append({
                    'from': frm.group(1).strip() if frm else '',
                    'date': date.group(1).strip() if date else '',
                    'message': msg.group(1).strip() if msg else text,
                })
    return sms_by_iban


def map_mails_to_users(mails_raw, users):
    """Map emails to user IBANs using To: header name."""
    name_to_iban = {}
    for u in users:
        name_to_iban[f"{u['first_name']} {u['last_name']}"] = u['iban']

    mails_by_iban = defaultdict(list)
    for m in mails_raw:
        text = m['mail']
        to_match = re.search(r'To: "(.+?)"', text)
        if to_match:
            name = to_match.group(1)
            iban = name_to_iban.get(name)
            if iban:
                subj = re.search(r'Subject: (.+)', text)
                frm = re.search(r'From: (.+)', text)
                date = re.search(r'Date: (.+)', text)
                body_text = re.sub(r'<[^>]+>', ' ', text)
                body_text = re.sub(r'\s+', ' ', body_text)[:400]
                mails_by_iban[iban].append({
                    'from': frm.group(1).strip() if frm else '',
                    'subject': subj.group(1).strip() if subj else '',
                    'date': date.group(1).strip() if date else '',
                    'body_snippet': body_text,
                })
    return mails_by_iban


def map_audio_to_users(audio_transcripts, users):
    """Map audio transcripts to user IBANs using filename usernames."""
    name_to_iban = {}
    for u in users:
        key = unicodedata.normalize('NFC', f"{u['first_name']}_{u['last_name']}".lower())
        name_to_iban[key] = u['iban']

    audio_by_iban = defaultdict(list)
    for username, records in audio_transcripts.items():
        normalized = unicodedata.normalize('NFC', username.lower())
        iban = name_to_iban.get(normalized)
        if not iban:
            first = normalized.split('_')[0]
            for key, val in name_to_iban.items():
                if key.startswith(first):
                    iban = val
                    break
        if iban:
            audio_by_iban[iban].extend(records)
    return audio_by_iban


# ═══════════════════════════════════════════════════════════════
# STEP 4 — BUILD USER DOSSIERS
# ═══════════════════════════════════════════════════════════════

def build_dossiers(users, df, locations, sms_by_iban, mails_by_iban, audio_by_iban):
    """Build per-user dossiers with profile, comms timeline, and all transactions."""
    iban_to_user = {}
    for u in users:
        iban_to_user[u['iban']] = u

    loc_by_biotag = defaultdict(list)
    for loc in locations:
        loc_by_biotag[loc['biotag']].append(loc)
    for bt in loc_by_biotag:
        loc_by_biotag[bt].sort(key=lambda x: x['timestamp'])

    # Map IBAN → biotag from transactions
    iban_to_biotag = {}
    for _, row in df.iterrows():
        sid = str(row.get('sender_id', ''))
        if is_biotag(sid) and pd.notna(row.get('sender_iban')):
            iban_to_biotag[row['sender_iban']] = sid

    dossiers = {}
    for u in users:
        iban = u['iban']
        comms = []
        for sms in sms_by_iban.get(iban, []):
            cat = classify_comm(sms.get('message', '') + ' ' + sms.get('from', ''))
            comms.append({
                'type': 'SMS', 'category': cat,
                'date': sms['date'], 'from': sms['from'],
                'content': sms['message'][:200],
            })
        for mail in mails_by_iban.get(iban, []):
            cat = classify_comm(
                mail.get('subject', '') + ' ' + mail.get('from', '') +
                ' ' + mail.get('body_snippet', ''))
            comms.append({
                'type': 'EMAIL', 'category': cat,
                'date': mail['date'], 'from': mail['from'],
                'subject': mail.get('subject', ''),
            })
        for audio in audio_by_iban.get(iban, []):
            cat = classify_comm(audio['transcript'])
            comms.append({
                'type': 'PHONE_CALL', 'category': cat,
                'date': audio['timestamp'],
                'transcript': audio['transcript'][:300],
            })
        comms.sort(key=lambda c: c.get('date', ''))

        user_txs = df[df['sender_iban'] == iban].copy()
        biotag = iban_to_biotag.get(iban)

        dossiers[iban] = {
            'user': u,
            'comms': comms,
            'transactions': user_txs,
            'biotag': biotag,
        }

    return dossiers, iban_to_user, loc_by_biotag


# ═══════════════════════════════════════════════════════════════
# STEP 5 — FORMAT DOSSIER FOR LLM
# ═══════════════════════════════════════════════════════════════

def format_dossier(dossier, iban_to_user, loc_by_biotag):
    """Format a user's complete dossier into LLM-readable text."""
    u = dossier['user']
    comms = dossier['comms']
    txs = dossier['transactions']
    biotag = dossier['biotag']
    home_lat = float(u['residence']['lat'])
    home_lng = float(u['residence']['lng'])

    lines = []

    # ── Profile ──
    lines.append(f"=== USER: {u['first_name']} {u['last_name']} ===")
    lines.append(f"Age: {2087 - u['birth_year']} | Job: {u['job']} | "
                 f"Annual salary: EUR {u['salary']:,} | Home: {u['residence']['city']}")
    lines.append(f"Profile: {u.get('description', 'N/A')}")
    lines.append("")

    # ── Communications ──
    phishing = [c for c in comms if c['category'] == 'PHISHING']
    normal = [c for c in comms if c['category'] == 'NORMAL']
    lines.append(f"COMMUNICATIONS: {len(comms)} total "
                 f"({len(phishing)} suspicious/phishing, {len(normal)} normal)")

    if phishing:
        lines.append("!! SUSPICIOUS/PHISHING COMMUNICATIONS:")
        for c in phishing:
            if c['type'] == 'SMS':
                lines.append(f"  [{c['date']}] SMS from {c['from']}: {c['content']}")
            elif c['type'] == 'EMAIL':
                lines.append(f"  [{c['date']}] EMAIL from {c['from']} -- {c.get('subject', '')}")
            elif c['type'] == 'PHONE_CALL':
                lines.append(f"  [{c['date']}] PHONE CALL: {c['transcript']}")

    if normal:
        lines.append(f"Normal messages: {len(normal)} (legitimate reminders, deliveries, etc.)")
    lines.append("")

    # ── Transaction statistics ──
    user_mean = 0
    user_std = 1
    if len(txs) > 0:
        amounts = pd.to_numeric(txs['amount'], errors='coerce')
        user_mean = amounts.mean()
        user_std = amounts.std() if amounts.std() > 0 else 1
        user_median = amounts.median()
        lines.append(f"TRANSACTION STATS: {len(txs)} outgoing | "
                     f"Mean: EUR {user_mean:.2f} | Median: EUR {user_median:.2f} | "
                     f"Std: EUR {user_std:.2f} | Range: EUR {amounts.min():.2f}-EUR {amounts.max():.2f}")
        type_counts = txs['transaction_type'].value_counts()
        lines.append(f"Types: {dict(type_counts)}")
    lines.append("")

    # ── Phishing dates for temporal correlation ──
    phishing_dates = []
    for c in phishing:
        dt = safe_ts(c.get('date', ''))
        if dt is not None:
            phishing_dates.append(dt)

    # ── All transactions ──
    lines.append("ALL TRANSACTIONS (chronological):")
    locs = loc_by_biotag.get(biotag, []) if biotag else []
    loc_timestamps = [l['timestamp'] for l in locs]

    for _, tx in txs.sort_values('timestamp').iterrows():
        amount = float(tx['amount'])
        ts = tx['timestamp']
        desc = tx.get('description', '')
        if pd.isna(desc) or desc == '':
            desc = ''
        location = tx.get('location', '')
        if pd.isna(location):
            location = ''
        balance = tx.get('balance_after', '')
        method = tx.get('payment_method', '')
        if pd.isna(method):
            method = ''

        # GPS distance
        dist = np.nan
        gps_city = ''
        if locs:
            idx = bisect_left(loc_timestamps, ts)
            cands = []
            if idx < len(locs):
                cands.append(locs[idx])
            if idx > 0:
                cands.append(locs[idx - 1])
            if cands:
                best = min(cands, key=lambda c: abs(
                    (safe_ts(c['timestamp']) - safe_ts(ts)).total_seconds()))
                dist = haversine_km(best['lat'], best['lng'], home_lat, home_lng)
                gps_city = best.get('city', '')

        # Flags
        flags = []
        if len(txs) > 1 and user_std > 0:
            zscore = (amount - user_mean) / user_std
            if abs(zscore) > 2:
                flags.append(f"UNUSUAL_AMOUNT({zscore:+.1f}s)")

        if pd.notna(dist):
            if dist > 500:
                flags.append(f"VERY_FAR({dist:.0f}km)")
            elif dist > 100:
                flags.append(f"FAR({dist:.0f}km)")

        if phishing_dates:
            tx_dt = safe_ts(ts)
            if tx_dt:
                recent = [p for p in phishing_dates
                          if p < tx_dt and (tx_dt - p).days <= 14]
                if recent:
                    days = (tx_dt - max(recent)).days
                    flags.append(f"PHISHING_BEFORE({days}d)")

        # Recipient info
        recip_iban = tx.get('recipient_iban', '')
        recip_info = ''
        if pd.notna(recip_iban) and recip_iban:
            recip_user = iban_to_user.get(recip_iban)
            if recip_user:
                recip_info = (f" -> {recip_user['first_name']} {recip_user['last_name']} "
                              f"({recip_user['job']})")
            else:
                recip_info = f" -> external({recip_iban[:12]}...)"

        # Build line
        line = f"  [{ts}] {tx['transaction_id']}: {tx.get('transaction_type', '')} EUR {amount:.2f}"
        if location:
            line += f" at {location}"
        if gps_city and gps_city != location:
            line += f" (GPS:{gps_city})"
        if method:
            line += f" [{method}]"
        line += f" bal=EUR {balance}"
        if desc:
            line += f' "{desc}"'
        if recip_info:
            line += recip_info
        if flags:
            line += " !! " + " | ".join(flags)

        lines.append(line)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# STEP 6 — LLM ANALYSIS
# ═══════════════════════════════════════════════════════════════

def generate_session_id():
    team = os.getenv('TEAM_NAME', 'team').replace(' ', '-')
    return f"{team}-{ulid.new().str}"


def analyze_user_llm(dossier, model, session_id, iban_to_user, loc_by_biotag):
    """Send a user's complete dossier to the LLM for fraud analysis."""
    handler = CallbackHandler()
    text = format_dossier(dossier, iban_to_user, loc_by_biotag)

    n_tx = len(dossier['transactions'])
    n_phish = sum(1 for c in dossier['comms'] if c['category'] == 'PHISHING')
    prompt = (f"Analyze this user's complete dossier ({n_tx} transactions, "
              f"{n_phish} phishing communications).\n"
              f"Return ONLY a JSON array of transaction_ids that are TRUE FRAUD.\n\n{text}")

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = model.invoke(messages, config={
        "callbacks": [handler],
        "metadata": {"langfuse_session_id": session_id},
    })

    content = response.content.strip()
    match = re.search(r'\[.*?\]', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []
    return []


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_folder(folder):
    print(f"\n{'=' * 60}")
    print(f" Processing: {folder}")
    print(f"{'=' * 60}\n")

    # Step 1: Load
    print("[1/5] Loading data...")
    df, users, locations, mails, sms, audio_files = load_data(folder)
    print(f"      tx={len(df)} users={len(users)} locs={len(locations)} "
          f"mails={len(mails)} sms={len(sms)} audio={len(audio_files)}")

    # Step 2: Transcribe audio
    print("[2/5] Audio transcription...")
    audio_transcripts = transcribe_audio(audio_files, folder)
    if audio_transcripts:
        total = sum(len(v) for v in audio_transcripts.values())
        print(f"      Transcribed {total} audio files for {len(audio_transcripts)} users")
    else:
        print("      No audio files")

    # Step 3: Map communications to users & build dossiers
    print("[3/5] Building user dossiers...")
    sms_by_iban = map_sms_to_users(sms, users)
    mails_by_iban = map_mails_to_users(mails, users)
    audio_by_iban = map_audio_to_users(audio_transcripts, users)

    dossiers, iban_to_user, loc_by_biotag = build_dossiers(
        users, df, locations, sms_by_iban, mails_by_iban, audio_by_iban)

    for iban, d in dossiers.items():
        u = d['user']
        n_tx = len(d['transactions'])
        n_phish = sum(1 for c in d['comms'] if c['category'] == 'PHISHING')
        n_calls = sum(1 for c in d['comms'] if c['type'] == 'PHONE_CALL')
        print(f"      {u['first_name']} {u['last_name']}: {n_tx} txs, "
              f"{n_phish} phishing, {n_calls} calls")

    # Step 4: LLM analysis
    print("[4/5] LLM fraud analysis per user...")
    model_id = os.getenv('REMOTE_MODEL', 'gpt-4o-mini')
    max_workers = int(os.getenv('MAX_CONCURRENCY', '8'))

    model = ChatOpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url='https://openrouter.ai/api/v1',
        model=model_id, temperature=0, max_tokens=4000,
    )
    langfuse_client = Langfuse(
        public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
        secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
        host=os.getenv('LANGFUSE_HOST', 'https://challenges.reply.com/langfuse'),
    )

    session_id = generate_session_id()
    print(f"      Session: {session_id}")
    print(f"      Model: {model_id} | Workers: {max_workers}")

    all_fraud_ids = []
    valid_tx_ids = set(df['transaction_id'])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for iban, dossier in dossiers.items():
            if len(dossier['transactions']) == 0:
                continue
            futures[executor.submit(
                analyze_user_llm, dossier, model, session_id,
                iban_to_user, loc_by_biotag
            )] = f"{dossier['user']['first_name']} {dossier['user']['last_name']}"

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="      Users", unit="user"):
            name = futures[future]
            try:
                fraud_ids = future.result()
                valid = [t for t in fraud_ids if t in valid_tx_ids]
                if valid:
                    print(f"        {name}: {len(valid)} fraud")
                all_fraud_ids.extend(valid)
            except Exception as e:
                print(f"        {name}: ERROR - {e}")

    langfuse_client.flush()

    all_fraud_ids = sorted(set(all_fraud_ids))

    # Step 5: Save results
    print(f"[5/5] Saving results...")
    out = os.path.join(folder, 'fraud_transactions.txt')
    with open(out, 'w') as f:
        for tid in all_fraud_ids:
            f.write(tid + '\n')

    print(f"      {len(all_fraud_ids)} fraud transactions -> {out}")
    return all_fraud_ids


def main():
    load_dotenv()
    if len(sys.argv) < 2:
        print('Usage: python main.py "folder1" ["folder2"] ...')
        sys.exit(1)

    folders = sys.argv[1:]
    results = {}
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Skipping '{folder}' - not found.")
            continue
        results[folder] = process_folder(folder)

    print(f"\n{'=' * 60}")
    print(f" SUMMARY")
    print(f"{'=' * 60}")
    for folder, ids in results.items():
        print(f"  {folder}: {len(ids)} fraud transactions")
    print(f"  TOTAL: {sum(len(v) for v in results.values())}")


if __name__ == '__main__':
    main()
