"""Capture Playwright des écrans clés du dashboard (documentation / portfolio).

Usage : streamlit doit tourner. Puis :
    PYTHONPATH=src .venv/bin/python scripts/capture_screens.py --url http://localhost:8533
"""

from __future__ import annotations

import argparse
import re

from playwright.sync_api import sync_playwright

from ciel_tranquille.config import get_settings

OUT = get_settings().output_path / "captures"


def _settle(page, ms: int = 2500) -> None:
    """Attend la fin du rendu Streamlit (disparition du spinner « Running »)."""
    try:
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=15000)
        page.wait_for_load_state("networkidle", timeout=15000)
        # Attend que le widget de statut (Running…) disparaisse.
        page.wait_for_function(
            """() => {
                const w = document.querySelector("[data-testid='stStatusWidget']");
                return !w || w.offsetParent === null;
            }""",
            timeout=20000,
        )
    except Exception:
        pass
    page.wait_for_timeout(ms)


def _goto_page(page, label: str) -> bool:
    """Clique le lien de navigation latérale correspondant au libellé."""
    try:
        link = page.get_by_role("link", name=re.compile(label, re.I)).first
        link.click(timeout=8000)
        _settle(page)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ! navigation '{label}' échouée : {exc}")
        return False


def capture(url: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # --- Desktop ---
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        print("→ Accueil (desktop)")
        page.goto(url, wait_until="domcontentloaded")
        _settle(page)
        page.screenshot(path=str(OUT / "accueil-desktop.png"), full_page=True)

        for label, fname in [
            ("Carte", "carte-zones-bruyantes.png"),
            ("Historique", "historique-bruit.png"),
            ("Prévision", "prevision-modele.png"),
            (r"Mod", "modeles-comparaison.png"),
        ]:
            print(f"→ {fname}")
            if _goto_page(page, label):
                page.screenshot(path=str(OUT / fname), full_page=True)
        ctx.close()

        # --- Mobile (responsive) ---
        mctx = browser.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True
        )
        mpage = mctx.new_page()
        print("→ Accueil (mobile)")
        mpage.goto(url, wait_until="domcontentloaded")
        _settle(mpage)
        mpage.screenshot(path=str(OUT / "accueil-mobile.png"), full_page=True)
        mctx.close()

        browser.close()
    print(f"✅ Captures écrites dans {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8533")
    capture(ap.parse_args().url)
