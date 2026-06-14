"""Generate a fresh shared access code, hash it for git, email the plaintext to admins.

Reads SMTP + admin secrets from environment (GitHub Actions secrets).
Writes data/access_code.json with the SHA-256 hash + rotation/expiry timestamps.
The plaintext code never lands in git.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
import smtplib
import string
import sys
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "access_code.json"
TTL_DAYS = int(os.environ.get("CDC_CODE_TTL_DAYS", "7"))


def generate_code(length: int = 10) -> str:
    alphabet = string.ascii_uppercase + string.digits
    # Drop visually ambiguous characters to make the code easy to type.
    alphabet = "".join(c for c in alphabet if c not in "0O1IL")
    return "".join(secrets.choice(alphabet) for _ in range(length))


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def write_hash_file(code: str) -> Path:
    now = dt.datetime.utcnow()
    payload = {
        "hash": hash_code(code),
        "rotated_at": now.isoformat(timespec="seconds") + "Z",
        "expires_at": (now + dt.timedelta(days=TTL_DAYS)).isoformat(timespec="seconds") + "Z",
        "algorithm": "sha256",
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return TARGET


def send_email(code: str) -> tuple[bool, str]:
    host = os.environ.get("CDC_SMTP_HOST", "").strip()
    user = os.environ.get("CDC_SMTP_USER", "").strip()
    pwd = os.environ.get("CDC_SMTP_PASS", "").strip()
    sender = os.environ.get("CDC_SMTP_FROM", "").strip() or user
    port_raw = os.environ.get("CDC_SMTP_PORT", "").strip()
    admins_raw = os.environ.get("CDC_ADMIN_EMAILS", "").strip()
    admins = [
        a.strip() for a in admins_raw.replace(";", ",").split(",") if a.strip()
    ]

    if not (host and user and pwd and sender and admins):
        return False, "SMTP or CDC_ADMIN_EMAILS not configured - skipping email"

    body = (
        "Bonjour,\n\n"
        f"Le code d'acces a la plateforme CDC LAUNCHPAD a ete renouvele.\n\n"
        f"  Nouveau code : {code}\n"
        f"  Valide jusqu'au : {(dt.datetime.utcnow() + dt.timedelta(days=TTL_DAYS)):%d %b %Y %H:%M} UTC\n\n"
        "Communiquez ce code aux membres autorises uniquement.\n"
        "L'ancien code ne fonctionne plus.\n\n"
        "-- CDC Tunisie - rotation automatique"
    )
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = "CDC LAUNCHPAD - rotation du code d'acces"
    msg["From"] = sender
    msg["To"] = ", ".join(admins)

    ports_to_try: list[tuple[int, str]] = []
    if port_raw:
        try:
            p = int(port_raw)
            ports_to_try.append((p, "ssl" if p == 465 else "starttls"))
        except ValueError:
            pass
    if not ports_to_try:
        ports_to_try = [(465, "ssl"), (587, "starttls")]

    last_err = ""
    for port, mode in ports_to_try:
        try:
            if mode == "ssl":
                with smtplib.SMTP_SSL(host, port, timeout=15) as smtp:
                    smtp.login(user, pwd)
                    smtp.sendmail(sender, admins, msg.as_string())
            else:
                with smtplib.SMTP(host, port, timeout=15) as smtp:
                    smtp.ehlo(); smtp.starttls(); smtp.ehlo()
                    smtp.login(user, pwd)
                    smtp.sendmail(sender, admins, msg.as_string())
            return True, f"sent via {host}:{port}/{mode} to {len(admins)} admin(s)"
        except Exception as exc:
            last_err = f"{host}:{port}/{mode} -> {type(exc).__name__}: {exc}"
            continue
    return False, f"all SMTP attempts failed - {last_err}"


def main() -> int:
    code = generate_code(10)
    target = write_hash_file(code)
    print(f"[rotate_code] wrote {target} (hash only)")

    ok, detail = send_email(code)
    if ok:
        print(f"[rotate_code] email {detail}")
    else:
        # Falls through to the workflow logs so the admin can still read the
        # plaintext when SMTP is misconfigured. Do NOT echo to public CI logs
        # if your repo is public - turn off the print or set the repo to private.
        print(f"[rotate_code] WARN: {detail}")
        print("=" * 60)
        print(f"PLAINTEXT CODE (visible only to admins with workflow log access):")
        print(f"  {code}")
        print(f"  expires in {TTL_DAYS} days")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
