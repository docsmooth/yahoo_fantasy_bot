#!/usr/bin/env python3
"""Isolate which step of the Yahoo OAuth + Fantasy read flow actually fails.

Yahoo gates the Fantasy Sports API behind a manual approval process
(https://sports.yahoo.com/developer/access/). An unapproved client still
completes OAuth perfectly -- it just gets a token with no 'scope' field, and
every /fantasy/v2 endpoint answers 403 "This application is not authorized to
perform this action." Step 4 below is what tells the two apart.

Uses only the stdlib so it runs in any interpreter. Never prints secrets.

  python3 yahoo_oauth_probe.py --oauth-file oauth2.json --callback-uri https://127.0.0.1/yahoobot/
  python3 yahoo_oauth_probe.py --key <id> --secret <secret> --callback-uri oob
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

AUTH = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN = "https://api.login.yahoo.com/oauth2/get_token"
FANTASY = "https://fantasysports.yahooapis.com/fantasy/v2"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def get(url, headers=None, follow=True):
    opener = urllib.request.build_opener() if follow else \
        urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers=headers or {})
    try:
        r = opener.open(req, timeout=30)
        return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def post(url, data, headers):
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode(data).encode(), headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def basic(key, secret):
    blob = base64.b64encode(f"{key}:{secret}".encode()).decode()
    return {"Authorization": "Basic " + blob,
            "Content-Type": "application/x-www-form-urlencoded"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oauth-file")
    p.add_argument("--key")
    p.add_argument("--secret")
    p.add_argument("--callback-uri", default="oob")
    p.add_argument("--write", metavar="PATH",
                   help="on success, write a yahoo_oauth-"
                        "compatible credentials file here")
    args = p.parse_args()

    key, secret = args.key, args.secret
    if args.oauth_file:
        creds = json.load(open(args.oauth_file, encoding="utf-8"))
        key = key or creds.get("consumer_key")
        secret = secret or creds.get("consumer_secret")
    if not key or not secret:
        sys.exit("Need --key/--secret or an --oauth-file containing them.")
    print(f"client_id: {key[:12]}... ({len(key)} chars), "
          f"secret: {len(secret)} chars")

    print("\n== Step 1: are the client credentials accepted? ==")
    code, body = post(TOKEN, {"code": "probe", "grant_type":
                              "authorization_code",
                              "redirect_uri": args.callback_uri},
                      basic(key, secret))
    text = body.decode(errors="replace")
    if "INVALID_AUTHORIZATION_CODE" in text:
        print("PASS - Yahoo authenticated the app; only the fake code failed.")
    else:
        print(f"FAIL - HTTP {code}: {text[:300]}")
        print("      The consumer key/secret pair is wrong for this app.")
        return

    print("\n== Step 2: is the callback URI registered? ==")
    q = urllib.parse.urlencode({"client_id": key,
                                "redirect_uri": args.callback_uri,
                                "response_type": "code"})
    url = f"{AUTH}?{q}"
    code, headers, _ = get(url, {"User-Agent": "Mozilla/5.0"}, follow=False)
    loc = headers.get("Location", "")
    if "oauth2/error" in loc:
        err = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
        print(f"FAIL - Yahoo rejected it: "
              f"{err.get('error_description', ['?'])[0]}")
        print(f"      Registered URI must match {args.callback_uri!r} exactly "
              f"(scheme, host, path, trailing slash).")
        return
    print(f"PASS - Yahoo accepted redirect_uri {args.callback_uri!r}.")

    print("\n== Step 3: exchange an authorization code ==")
    print("Open this URL, sign in, approve, then copy the value of the "
          "'code=' parameter out of the address bar.")
    print("(The page will fail to load -- nothing listens on 127.0.0.1. "
          "That is expected; the code is still in the URL.)\n")
    print(url + "\n")
    t0 = time.time()
    verifier = input("code= ").strip()
    elapsed = time.time() - t0
    if not verifier:
        sys.exit("No code supplied.")
    if verifier.startswith("http"):
        u = urllib.parse.urlparse(verifier)
        landed = urllib.parse.urlunparse(
            (u.scheme, u.netloc, u.path, "", "", ""))
        verifier = urllib.parse.parse_qs(u.query).get("code", [""])[0]
        print(f"(extracted code from the pasted URL)")
        if landed != args.callback_uri:
            print(f"  ERROR: the browser landed on {landed!r} but this run "
                  f"was started with --callback-uri {args.callback_uri!r}.")
            print("  Yahoo binds the code to the authorize-time redirect_uri "
                  "and rejects the exchange with the misleading error "
                  "'INVALID_AUTHORIZATION_CODE'.")
            print(f"  Rerun with: --callback-uri '{landed}'")
            sys.exit(1)
    verifier = verifier.strip().strip("#/&?").strip()
    print(f"code: {len(verifier)} chars, charset_ok="
          f"{verifier.isalnum()}, pasted {elapsed:.0f}s after the URL "
          f"was printed")
    if elapsed > 60:
        print("  WARNING: Yahoo authorization codes expire ~60s after "
              "they are issued. This is very likely the failure.")
    if not verifier.isalnum():
        bad = sorted({c for c in verifier if not c.isalnum()})
        print(f"  WARNING: non-alphanumeric characters in code: {bad} "
              f"-- you probably copied surrounding text.")

    t1 = time.time()
    code, body = post(TOKEN, {"code": verifier, "grant_type":
                              "authorization_code",
                              "redirect_uri": args.callback_uri},
                      basic(key, secret))
    text = body.decode(errors="replace")
    if code != 200:
        print(f"FAIL - HTTP {code}: {text[:400]}")
        print(f"      (exchange issued {time.time() - t1:.1f}s after you "
              f"pressed enter)")
        if "INVALID_AUTHORIZATION_CODE" in text:
            print("      Codes are single-use and short-lived. Rerun, and "
                  "have the browser tab already open+signed in so the "
                  "copy/paste takes only a few seconds.")
        return
    tokens = json.loads(text)
    print(f"PASS - got {sorted(tokens)} "
          f"(access_token {len(tokens['access_token'])} chars)")
    scope = tokens.get("scope")
    if scope:
        print(f"       scope: {scope!r}")
    else:
        print("       scope: ABSENT -- this client has no Fantasy Sports "
              "entitlement.\n"
              "       Registering the app and ticking 'Fantasy Sports - "
              "Read' does not grant it;\n"
              "       approval is provisioned per Client ID via "
              "https://sports.yahoo.com/developer/access/")

    if args.write:
        out = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": tokens["token_type"],
            "token_time": time.time(),
            "consumer_key": key,
            "consumer_secret": secret,
            "callback_uri": args.callback_uri,
        }
        with open(args.write, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=4, sort_keys=True)
        os.chmod(args.write, 0o600)
        print(f"       wrote credentials to {args.write} (mode 0600)")

    print("\n== Step 4: does the Fantasy Sports API accept the token? ==")
    hdr = {"Authorization": "Bearer " + tokens["access_token"]}
    for path in ("users;use_login=1/games",
                 "users;use_login=1/games;game_codes=nhl/leagues"):
        status, _, raw = get(f"{FANTASY}/{path}?format=json", hdr)
        snippet = " ".join(raw.decode(errors="replace").split())[:400]
        verdict = "PASS" if status == 200 else "FAIL"
        print(f"{verdict} [{status}] {path}\n      {snippet}\n")


if __name__ == "__main__":
    main()
