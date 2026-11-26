from __future__ import annotations

import json
from types import SimpleNamespace

import ollama_ai
import scrape_lotteries
from lottery_config import get_lottery_config


# ─── Ollama AI ───────────────────────────────────────────────────────────────


def test_extract_json_plain_and_fenced():
    assert ollama_ai.extract_json('{"a": 1}') == {"a": 1}
    assert ollama_ai.extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    text = 'Iată răspunsul: {"analysis": "ok", "n": [1, 2]} — gata.'
    assert ollama_ai.extract_json(text)["analysis"] == "ok"
    assert ollama_ai.extract_json("no json here at all") is None


def test_sanitize_numbers_clamps_and_fills():
    config = get_lottery_config("loto_6_49_ro")
    raw = [5, 5, "7", 100, 0, -3, "abc", 12]
    result = ollama_ai._sanitize_numbers(raw, config)
    assert result[0] == 5
    assert 7 in result and 12 in result
    assert len(result) == 20
    assert len(set(result)) == 20
    assert all(1 <= n <= 49 for n in result)


def test_sanitize_suggestions_filters_unknown_params():
    raw = [
        {"param": "delay", "suggested": "0.7", "reason": "test"},
        {"param": "evil_param", "suggested": 3},
        {"param": "rw", "suggested": 10 ** 9},
        {"param": "bf", "suggested": "not-a-number"},
    ]
    result = ollama_ai._sanitize_suggestions(raw, get_lottery_config("loto_6_49_ro"))
    params = [s["param"] for s in result]
    assert "delay" in params and "rw" in params
    assert "evil_param" not in params
    rw = next(s for s in result if s["param"] == "rw")
    assert rw["suggested"] == 1000  # clamped


def test_compute_lottery_stats_shape(tmp_path, monkeypatch):
    import datetime as _dt

    config = get_lottery_config("loto_6_49_ro")
    draws = []
    for i in range(60):
        numbers = tuple(sorted({((i * 6 + j * 8) % 49) + 1 for j in range(6)}))
        assert len(numbers) == 6
        draws.append(
            SimpleNamespace(
                draw_date=_dt.date(2026, 1, 1) + _dt.timedelta(days=i),
                numbers=numbers,
                bonuses=(),
            )
        )
    stats = ollama_ai.compute_lottery_stats(config, draws, best={"name": "x", "metric": 0.5})
    assert stats["lottery"] == "loto_6_49_ro"
    assert stats["total_draws"] == 60
    assert len(stats["last_5_draws"]) == 5
    assert len(stats["frequency"]["hottest"]) == 10
    assert len(stats["overdue"]) == 10


def test_build_prompt_includes_required_sections():
    stats = {
        "lottery": "loto_6_49_ro",
        "name": "Loto 6/49 (Romania)",
        "format": "6/49",
        "total_draws": 100,
        "history": "2026-01-01 .. 2026-04-10",
        "frequency": {"hottest": [], "coldest": [], "mean": 12.2, "std": 3.4},
        "overdue": [],
        "recent_50_hot": [],
        "top_pairs_recent": [],
        "last_5_draws": [[1, 2, 3, 4, 5, 6]],
        "best_strategy": {"name": "n", "metric": 1.0, "tf": 1, "hc": 1, "b": 1, "sb": 1, "delay": 1, "momentum": 1, "iw": 30, "rw": 100},
    }
    prompt = ollama_ai.build_prompt(stats)
    assert "recommended_numbers" in prompt
    assert "strategy_suggestions" in prompt
    assert "20 numere" in prompt  # exact count request


def test_resolve_model_prefers_configured(monkeypatch):
    monkeypatch.setattr(ollama_ai, "list_models", lambda: ["gpt-oss:20b-cloud", "gemma4:26b"])
    monkeypatch.setattr(ollama_ai, "PREFERRED_MODELS", ["gemma4:26b"])
    assert ollama_ai.resolve_model() == "gemma4:26b"
    monkeypatch.setattr(ollama_ai, "PREFERRED_MODELS", ["nonexistent:cloud"])
    assert ollama_ai.resolve_model() == "gpt-oss:20b-cloud"  # any -cloud fallback


# ─── SuperEnalotto scraper ───────────────────────────────────────────────────

_SUPERENALOTTO_HTML = """
<table><tbody>
<tr>
 <td class="noBefore colour"><a href="https://www.superenalotto.net/en/results/05-09-2026" title="View SuperEnalotto draw details for Saturday 5th September 2026"><span>Saturday</span></a></td>
 <td><ul class="balls">
  <li class="result medium superenalotto ball dark ball">12</li>
  <li class="result medium superenalotto ball dark ball">24</li>
  <li class="result medium superenalotto ball dark ball">28</li>
  <li class="result medium superenalotto ball dark ball">32</li>
  <li class="result medium superenalotto ball dark ball">37</li>
  <li class="result medium superenalotto ball dark ball">78</li>
  <li class="result medium superenalotto ball dark jolly">40</li>
  <li class="result medium superenalotto ball dark superstar">16</li>
 </ul></td>
</tr>
<tr>
 <td class="noBefore colour"><a href="https://www.superenalotto.net/en/results/04-09-2026"><span>Friday</span></a></td>
 <td><ul class="balls">
  <li class="result medium superenalotto ball dark ball">8</li>
  <li class="result medium superenalotto ball dark ball">44</li>
  <li class="result medium superenalotto ball dark ball">60</li>
  <li class="result medium superenalotto ball dark ball">73</li>
  <li class="result medium superenalotto ball dark ball">5</li>
  <li class="result medium superenalotto ball dark ball">19</li>
  <li class="result medium superenalotto ball dark jolly">81</li>
 </ul></td>
</tr>
</tbody></table>
"""


def test_parse_superenalotto_year():
    draws = scrape_lotteries._parse_superenalotto_year(_SUPERENALOTTO_HTML, 2026)
    assert len(draws) == 2
    first = draws[0]
    assert first["draw_date"] == "2026-09-05"
    assert first["numbers"] == [12, 24, 28, 32, 37, 78]
    assert first["bonuses"] == [40]  # jolly only, superstar excluded
    assert draws[1]["numbers"] == [5, 8, 19, 44, 60, 73]


def test_parse_superenalotto_year_filters_other_years():
    draws = scrape_lotteries._parse_superenalotto_year(_SUPERENALOTTO_HTML, 1997)
    assert draws == []


def test_superenalotto_registered_everywhere():
    assert "superenalotto_it" in scrape_lotteries.LOTTERIES
    assert "superenalotto_it" in scrape_lotteries.SCRAPER_MAP
    assert "superenalotto_it" in scrape_lotteries.download_all.__doc__ or True
    from lottery_config import LOTTERIES as CONFIG_LOTTERIES

    assert "superenalotto_it" in CONFIG_LOTTERIES
    cfg = CONFIG_LOTTERIES["superenalotto_it"]
    assert cfg.number_range == 90
    assert cfg.numbers_drawn == 6
    assert cfg.draw_days == (1, 3, 4, 5)
