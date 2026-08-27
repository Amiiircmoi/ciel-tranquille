"""Tests du gel de l'instantané : manifeste, somme de contrôle, immuabilité."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ciel_tranquille.ml import dataset


def _paires(n=40):
    return pd.DataFrame({
        "event_id": range(n),
        "station": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "airport": "CDG",
        "max_laeq": [70.0 + i % 10 for i in range(n)],
        "slant_km": [1.0 + (i % 7) for i in range(n)],
        "max_ts_unix": [1_787_600_000 + i * 60 for i in range(n)],
        "hour_utc": "2026-08-25T14",
    })


def test_checksum_changes_with_content(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _paires().to_parquet(a, index=False)
    _paires(41).to_parquet(b, index=False)
    assert dataset.sha256_file(a) != dataset.sha256_file(b)
    assert len(dataset.sha256_file(a)) == 64


def test_checksum_is_stable_across_reads(tmp_path):
    chemin = tmp_path / "a.parquet"
    _paires().to_parquet(chemin, index=False)
    assert dataset.sha256_file(chemin) == dataset.sha256_file(chemin)


def test_manifest_carries_what_makes_the_run_reproducible(settings, tmp_path, monkeypatch):
    """Fenêtre, volume, stations, taux, empreinte : les cinq preuves exigées."""
    monkeypatch.setattr(dataset, "build_dataset", lambda *a, **k: _paires())
    progress = {"totals": {"events": 50, "clean_pairs": 40}}
    settings.pairs_progress_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pairs_progress_path.write_text(json.dumps(progress), encoding="utf-8")

    manifeste = dataset.export_snapshot(settings, out_dir=tmp_path / "gel", label="essai")

    assert manifeste["n_paires"] == 40
    assert manifeste["taux_association"] == pytest.approx(0.8)
    assert manifeste["stations"] == ["A", "B"]
    assert manifeste["fenetre"]["premier_evenement_iso"] == "2026-08-24T19:33:20Z"
    assert len(manifeste["sha256"]) == 64
    assert "10.0 km" in manifeste["regle_association"]


def test_exported_file_matches_its_own_manifest(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "build_dataset", lambda *a, **k: _paires())
    manifeste = dataset.export_snapshot(settings, out_dir=tmp_path / "gel", label="essai")

    donnees = tmp_path / "gel" / "pairs_essai.parquet"
    ecrit = json.loads((tmp_path / "gel" / "pairs_essai.manifest.json").read_text(encoding="utf-8"))
    assert ecrit == manifeste
    assert dataset.sha256_file(donnees) == manifeste["sha256"]
    assert len(pd.read_parquet(donnees)) == manifeste["n_paires"]


def test_export_lands_outside_the_collection_directory(settings, tmp_path, monkeypatch):
    """Un instantané figé n'a rien à faire dans le répertoire que le poller écrit."""
    monkeypatch.setattr(dataset, "build_dataset", lambda *a, **k: _paires())
    dataset.export_snapshot(settings, out_dir=tmp_path / "gel", label="essai")
    assert not (settings.data_path / "pairs_essai.parquet").exists()
    assert (tmp_path / "gel" / "pairs_essai.parquet").exists()


def test_empty_dataset_is_a_loud_failure(settings, tmp_path, monkeypatch):
    """Zéro paire ne doit jamais produire un instantané vide d'apparence valide."""
    monkeypatch.setattr(dataset, "build_dataset", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="Aucune paire propre"):
        dataset.export_snapshot(settings, out_dir=tmp_path / "gel", label="vide")


def test_association_rate_is_reread_not_recomputed(settings, tmp_path, monkeypatch):
    """Un second calcul du dénominateur donnerait un second chiffre discordant."""
    monkeypatch.setattr(dataset, "build_dataset", lambda *a, **k: _paires())
    settings.pairs_progress_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pairs_progress_path.write_text(
        json.dumps({"totals": {"events": 100}}), encoding="utf-8"
    )
    manifeste = dataset.export_snapshot(settings, out_dir=tmp_path / "gel", label="essai")
    assert manifeste["n_evenements_denominateur"] == 100
    assert manifeste["taux_association"] == pytest.approx(0.4)


def test_missing_progress_file_leaves_the_rate_unknown(settings, tmp_path, monkeypatch):
    """Sans dénominateur, on écrit « inconnu » plutôt qu'un chiffre inventé."""
    monkeypatch.setattr(dataset, "build_dataset", lambda *a, **k: _paires())
    manifeste = dataset.export_snapshot(settings, out_dir=tmp_path / "gel", label="essai")
    assert manifeste["taux_association"] is None
    assert manifeste["n_paires"] == 40
