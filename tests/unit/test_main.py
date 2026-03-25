from __future__ import annotations


def test_app_imports() -> None:
    from product_fidelity_lab.main import app

    assert app.title == "Product Fidelity Lab"
