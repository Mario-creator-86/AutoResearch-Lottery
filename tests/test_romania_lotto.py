from __future__ import annotations

from lottery_config import get_lottery_config
from romania_lotto import _extract_available_years, parse_results_page
import server


def test_extract_available_years_from_official_form():
    html = """
    <form id="formSubmit">
      <select name="select-year">
        <option value="0">Selectati anul</option>
        <option value="2026" selected="selected">2026</option>
        <option value="2025">2025</option>
        <option value="2024">2024</option>
      </select>
    </form>
    """

    assert _extract_available_years(html) == [2024, 2025, 2026]


def test_parse_loto_6_49_result_block():
    html = """
    <div class="rezultate-extrageri-content resultDiv">
      <div class="info-rezultat">
        <div class="numere-extrase">
          <img src="/bile/46.png" />
          <img src="/bile/18.png" />
          <img src="/bile/5.png" />
          <img src="/bile/15.png" />
          <img src="/bile/41.png" />
          <img src="/bile/28.png" />
        </div>
        <div class="button-open-details"><p>Detalii castiguri la loto 6/49 din <span>26.03.2026</span></p></div>
      </div>
      <table class="results-table">
        <tbody>
          <tr><td>I</td><td>1</td><td>12.345,67</td><td>-</td></tr>
          <tr class="total"><td colspan="4">Fond total de castiguri: <strong>98.765,43</strong></td></tr>
        </tbody>
      </table>
    </div>
    <div class="rezultate-extrageri-content resultDiv">
      <div class="info-rezultat">
        <div class="button-open-details"><p>Detalii noroc din <span>26.03.2026</span></p></div>
      </div>
    </div>
    """

    results = parse_results_page("loto_6_49_ro", html)

    assert len(results) == 1
    assert results[0].draw_date.isoformat() == "2026-03-26"
    assert results[0].numbers == (5, 15, 18, 28, 41, 46)
    assert results[0].bonuses == ()
    assert results[0].tier1_winners == 1
    assert results[0].tier1_amount == 12345.67
    assert results[0].is_rollover is False
    assert results[0].total_pool == 98765.43


def test_parse_joker_result_block_uses_report_as_rollover_amount():
    html = """
    <div class="rezultate-extrageri-content resultDiv">
      <div class="info-rezultat">
        <div class="numere-extrase">
          <img src="/bile/29.png" />
          <img src="/bile/3.png" />
          <img src="/bile/17.png" />
          <img src="/bile/33.png" />
          <img src="/bile/7.png" />
          <img src="/bile/joker/8.png" />
        </div>
        <div class="button-open-details"><p>Detalii castiguri la joker din <span>30.01.2025</span></p></div>
      </div>
      <table class="results-table">
        <tbody>
          <tr><td>I (5/5+J)</td><td>REPORT</td><td>230.058,60</td><td>9.918.309,97</td></tr>
          <tr class="total"><td colspan="4">Fond total de castiguri: <strong>10.371.427,59</strong></td></tr>
        </tbody>
      </table>
    </div>
    """

    results = parse_results_page("joker_ro", html)

    assert len(results) == 1
    assert results[0].draw_date.isoformat() == "2025-01-30"
    assert results[0].numbers == (3, 7, 17, 29, 33)
    assert results[0].bonuses == (8,)
    assert results[0].tier1_winners == 0
    assert results[0].tier1_amount == 9918309.97
    assert results[0].is_rollover is True
    assert results[0].total_pool == 10371427.59


def test_parse_loto_5_40_result_block_with_six_drawn_numbers():
    html = """
    <div class="rezultate-extrageri-content resultDiv">
      <div class="info-rezultat">
        <div class="numere-extrase">
          <img src="/bile/22.png" />
          <img src="/bile/13.png" />
          <img src="/bile/40.png" />
          <img src="/bile/19.png" />
          <img src="/bile/6.png" />
          <img src="/bile/2.png" />
        </div>
        <div class="button-open-details"><p>Detalii castiguri la 5/40 din <span>26.03.2026</span></p></div>
      </div>
      <table class="results-table">
        <tbody>
          <tr><td>I (5/6*)</td><td>1</td><td>420.765,00</td><td>-</td></tr>
          <tr class="total"><td colspan="4">Fond total de castiguri: <strong>495.164,00</strong></td></tr>
        </tbody>
      </table>
    </div>
    """

    results = parse_results_page("loto_5_40_ro", html)

    assert len(results) == 1
    assert results[0].numbers == (2, 6, 13, 19, 22, 40)
    assert results[0].tier1_amount == 420765.0
    assert results[0].tier1_winners == 1


def test_romanian_lottery_configs_and_ticket_outcomes():
    loto_649 = get_lottery_config("loto_6_49_ro")
    loto_540 = get_lottery_config("loto_5_40_ro")
    joker = get_lottery_config("joker_ro")

    assert loto_649.number_range == 49
    assert loto_540.number_range == 40
    assert loto_540.numbers_drawn == 6
    assert joker.number_range == 45
    assert joker.bonus_range == 20

    assert server._evaluate_ticket_outcome("loto_6_49_ro", 3, 0, 6)["won"] is True
    assert server._evaluate_ticket_outcome("loto_6_49_ro", 2, 0, 6)["won"] is False
    assert server._evaluate_ticket_outcome("loto_5_40_ro", 4, 0, 5)["won"] is True
    assert server._evaluate_ticket_outcome("loto_5_40_ro", 3, 0, 5)["won"] is False
    assert server._evaluate_ticket_outcome("joker_ro", 1, 1, 5)["won"] is True
    assert server._evaluate_ticket_outcome("joker_ro", 3, 0, 5)["won"] is True
    assert server._evaluate_ticket_outcome("joker_ro", 2, 0, 5)["won"] is False
