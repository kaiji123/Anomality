import pandas as pd

from app.dashboard_utils import truncate_at_date, verdict


def test_truncate_never_after_cutoff():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-02-10", "2020-02-14", "2020-02-20"]), "x": [1, 2, 3]})
    out = truncate_at_date(df, "date", pd.Timestamp("2020-02-14"))
    assert (out["date"] <= pd.Timestamp("2020-02-14")).all()


def test_verdict_rule():
    assert verdict(0.1, 0.2) == "supported"
    assert verdict(-0.2, -0.01) == "contradicted"
    assert verdict(-0.1, 0.1) == "inconclusive"
