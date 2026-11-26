# Audit suplimentar al logicii — 19 septembrie 2026

## Continuare: cauzalitatea strategiilor și izolarea loteriilor

Verificare finală pentru această etapă: **233 teste trecute în 11,53 secunde**,
cu 19 cazuri suplimentare în `tests/test_causal_strategies.py`.

- **Informații viitoare în indicatorul jackpot:** pragul proxy și durata mediană
  a ciclurilor erau calculate pe întregul istoric; istoricul real folosea și
  câștigători din viitor pentru duratele medii. `jackpot_proxy.py` și
  `jackpot_data.py` folosesc acum numai observațiile disponibile până la fiecare
  pas. Testele compară indicatorii unui istoric scurt cu aceiași indicatori după
  adăugarea unor extrageri ulterioare.
- **Metadate zilnice folosite prea devreme:** la două extrageri în aceeași zi,
  indicatorul zilnic putea include informații de la extragerea a doua. Evaluarea
  istorică din `generic_backtest.py` folosește acum ultima zi anterioară când
  urmează încă o extragere în aceeași zi.
- **Zile lipsă în istoricul jackpot:** contorul nu avansa în zilele pentru care
  erau folosite date proxy. Acum contorul față de ultimul câștig cunoscut avansează
  și peste aceste înregistrări lipsă.
- **Context amestecat între loterii:** predicția live reutiliza primul context
  jackpot global, inclusiv pentru altă loterie sau un istoric actualizat.
  `auto_loop.py` furnizează contextul explicit fiecărei strategii și construiește
  contextul loteriei cerute și pentru walk-forward.
- **Ponderi globale instabile:** funcția de clasare construită păstrează propria
  copie a ponderilor de căutare. Predicția live nu mai modifică ponderile globale.
- **Pluginuri cu nume identice:** `ai_strategy_lab.py` indexează acum pluginurile
  prin perechea loterie–cheie. Construirea unei strategii încarcă pluginurile
  promovate pentru loteria cerută și elimină înregistrările retrase.

Proxy-ul a fost măsurat separat, fără scrierea datelor: aproximativ **0,19 secunde
pentru 17.030 de extrageri Multi Multi**. Fereastra de frecvențe este actualizată
incremental.

**Scorurile istorice salvate pentru strategiile care folosesc ciclul jackpot nu
au fost recalculate în această etapă.** Ele nu trebuie tratate ca rezultate ale
metodei corectate; este necesară reevaluarea. Corecția contaminării temporale nu
elimină problema distinctă a selectării parametrilor pe întregul istoric.

## Probleme reproduse și corectate

1. **Generarea live folosea un indice diferit de backtest.** După 15 extrageri,
   backtestul transmitea indicele 15 pentru următoarea predicție, dar generarea
   live transmitea 14. Aceasta afecta strategiile care folosesc întârzieri,
   frecvențe normalizate sau faza jackpotului. `auto_loop.py` transmite acum
   numărul extragerilor procesate. Testul compară contextul și predicția live
   cu predicția finală a backtestului pentru aceeași strategie.

2. **Validarea CSV verifica doar intervalele numerice.** Rândurile cu numere
   repetate, număr incorect de numere sau bonusuri lipsă puteau intra în calcule.
   `load_data.py` verifică acum cardinalitatea și unicitatea ambelor grupuri.
   Încărcarea filtrată folosește aceeași istorie validată și memorată în cache,
   eliminând parserul duplicat.

3. **Deduplicarea putea elimina o extragere distinctă.** Două extrageri din aceeași
   zi cu aceleași numere erau confundate, chiar dacă aveau ore și identificatori
   diferiți. Invers, repetarea unui rând cu numerele în altă ordine era păstrată.
   Cheia include acum metadatele extragerii și numerele sortate.

4. **Biletele puteau fi verificate față de extragerea greșită.** Când ID-ul sau ora
   cerută lipsea din istoric, serverul putea folosi ultima extragere a zilei.
   Agentul folosea poziția chiar și atunci când metadatele de oră indicau altceva.
   `draw_matching.py`, folosit de server și agent, caută explicit ID-ul sau ora.
   Dacă ora cerută nu corespunde istoricului cu ore cunoscute, rezultatul rămâne
   indisponibil. Poziția rămâne compatibilă cu istoricul fără ore.

5. **Istoricul gol bloca actualizarea rezultatelor.** Agentul ieșea înainte de
   încercarea de descărcare. Acum încearcă actualizarea și în acest caz.

## Verificare

- Problemele de validare, deduplicare, indice și asociere au fost reproduse prin
  teste care au eșuat înainte de corecții.
- 27 de cazuri noi față de suita de 187 de teste de la auditul anterior.
- Suita completă: **214 teste trecute în 11,63 secunde**.
- Verificare doar prin citire a celor 13 loterii locale: **55.692 extrageri**;
  numărul rândurilor încărcate a rămas același după corecția validatorului.
- Testele agentului izolează descărcările și jurnalul. Testele auto-loop izolează
  notificările externe și declanșarea validării walk-forward; motorul walk-forward
  este verificat separat prin testele de regresie.

## Limite ale verificării

- Istoricul local încărcat nu conține ore de extragere. Pentru două extrageri
  zilnice, asocierea prin poziție depinde în continuare de ordinea și completitudinea
  datelor sursă. Fără ID sau oră nu se poate identifica sigur o extragere lipsă
  din mijlocul unei zile.
- Nu au fost recalculate sau rescrise retroactiv rezultatele biletelor deja
  marcate ca verificate. Corecțiile se aplică evaluărilor efectuate de codul nou.
- Trecerea testelor confirmă comportamentele acoperite, nu un avantaj predictiv
  al strategiilor și nici absența tuturor defectelor din proiect.
