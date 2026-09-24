# Recenze plánu článku P5 — „Frozen LLM as a variation operator in evolutionary AutoML pipeline search“

Cíl: EvoLearn 2027 (EvoStar, LNCS, double-blind), deadline 1. 11. 2026.
Stav: infrastruktura hotová, pilot dokončen, Phase 3 před spuštěním (preregistrace).

---

## 1. Celkové hodnocení

**Verdikt: přijatelné s většími revizemi plánu před spuštěním Phase 3.**

Plán je dobře postaven: rovný počet skutečných evaluací jako primární osa, anytime křivky, preregistrace, předem deklarovaná podmínka pro vyvrácení H1 a — především — měřicí program (§4.7), který je nejcennější a nejméně pokrytou částí. Kritická/mechanismová framing sedí na EvoLearn a má precedens (EvoApplications 2025).

Hlavní rizika pro přijetí jsou tři a všechna jsou řešitelná před spuštěním:

1. **Negativní výsledek bez ekvivalenčního testu** — Wilcoxon přes 15 úloh nezamítne nulu při Δ ≈ −0.003, což je „absence of evidence“.
2. **H2 (structural narrowness) může být artefakt promptu** (worked examples), ne modelu.
3. **Prostor má jen 63 diskrétních struktur** — remíza s random search je v něm očekávatelná a novelty metriky jsou mechanicky nasycené.

Doporučená investice v pořadí: (a) statistický rámec pro H1, (b) promptová ablace, (c) druhý, větší prostor jako sekundární experiment (návrh v §5).

---

## 2. Silné stránky

- **Férový budget.** Jednotka = skutečná (necachovaná) evaluace, anytime křivky, společný start populace. Odzbrojuje nejčastější EC námitku.
- **Měřicí program.** Validita podle typu porušení, duplikáty vs. populace i historie, diverzita dvojím způsobem, strukturální novelty. Publikovatelné i při remíze.
- **Preregistrace a decision gate.** Výběr úloh nezávislý na výsledku, předem daná podmínka vyvrácení H1. Vzácné, recenzenti to ocení.
- **Kontrolní podmnožina hp-decisive úloh.** Rozumná obrana proti „rigged tasks“ (byť ne statistická, viz §3.7).
- **Infrastrukturní disciplína.** Hard cap, preflight, ledger, in-memory cache — správně, ale do článku jen zkráceně.

---

## 3. Rizika a doporučení (seřazeno podle závažnosti)

### 3.1 H1 potřebuje ekvivalenční test, ne test rozdílu
Wilcoxon signed-rank s n = 15 úlohami a Δ ≈ −0.003 nezamítne nulu prakticky nikdy. Recenzent řekne: „nenašli jste efekt, protože nemáte sílu ho najít“.

**Doporučení:**
- Přeformulovat H1 jako *ohraničení efektu*: TOST (two one-sided tests) s předem stanovenou mezí ekvivalence (např. ±0.5 pp accuracy), nebo minimálně „jakákoli výhoda LLM je s 95% bootstrap CI menší než X“.
- Mez ekvivalence zapsat do preregistrace.
- Reportovat velikosti efektu (per-task Δ s CI), ne jen p-hodnoty.

### 3.2 H2 může být způsobena promptem
Statický prefix obsahuje worked examples. Pokud LLM kopíruje jejich struktury, je „structural narrowness“ z části few-shot kotvení, ne vlastnost modelu. §8 to přiznává jako „prompt dependence“, ale recenzent to použije proti vám.

**Doporučení:**
- Ověřit, že struktury z worked examples nejsou při výpočtu novelty počítány jako „seen“ zdarma (resp. reportovat obě varianty).
- Přidat levnou **promptovou ablaci**: (i) bez worked examples, (ii) s explicitní instrukcí navrhovat nové struktury. Pokud narrowness přetrvá, H2 je robustní; pokud zmizí, je to stejně zajímavý výsledek.
- Do ablací přidat **teplotu vzorkování** — přímo ovlivňuje diverzitu, a je to laciné.

### 3.3 Velikost prostoru: 63 struktur
3 × 3 × 7 = 63 diskrétních struktur (pre/fs/est). Při populaci ~30 a ~30 generacích jsou všechny struktury „seen“ po několika generacích a podíl nových struktur je mechanicky nízký pro *každý* operátor. Zároveň: random search v 63 strukturách přirozeně remizuje, takže „bounded space“ v §8 je pro obecnost výsledku fatální.

**Doporučení:**
- Reportovat strukturální novelty **časově rozlišenou** (po generacích / po evaluacích), ne agregovanou přes běh.
- Reportovat pokrytí prostoru (podíl struktur navštívených do generace g) pro obě metody.
- Přidat **druhý, větší prostor jako sekundární experiment** (viz §5). Mění limitaci na výsledek: „mění se rozdíl LLM vs. random s velikostí prostoru?“

### 3.4 H4 pilot nepodporuje
Pilot: duplikáty vs. historie 0.13 (LLM) vs. 0.37 (random). To je opak H4 („LLM plýtvá budgetem způsobem, kterým random ne“).

**Doporučení:**
- H4 buď přeformulovat („LLM opakuje *koncentrovaně* v úzké oblasti, random opakuje *plošně*“) a doložit metrikou (např. entropie distribuce duplikátů), nebo H4 vypustit.
- **37 % duplikátů u náhodného operátoru je podezřelé.** Se spojitými HP by náhodná mutace neměla generovat klony. Pravděpodobné příčiny: mutace mění jen jeden gen a crossover konvergované populace produkuje kopie; nebo HP vzorkovány z hrubé mřížky. Vysvětlit design náhodného operátoru explicitně a případně opravit — jinak je baseline slabá v opačném směru, což podkopává férovost.

### 3.5 Dva budgety, ne jeden
§4.5: duplikáty jsou zdarma. H4: duplikáty plýtvají budgetem. Rozpor.

**Doporučení:**
- Rozlišit **budget návrhů** (API volání, generace) a **budget skutečných evaluací**.
- Reportovat obě osy: best-vs-genuine-evaluations (primární) *a* best-vs-proposals (sekundární; ukazuje reálnou cenu LLM operátoru včetně duplikátů a fallbacků).
- K otázce 1 plánu: čistší je vynutit **přesně stejný počet skutečných evaluací** (resample při duplikátu) jako primární design; různě dlouhé křivky reportovat sekundárně. Vyhnete se otázce „co je common budget, když se běhy liší délkou“.

### 3.6 Terminologie diverzity
§4.7 nazývá edit distance „structural“; §6 říká, že LLM má *vyšší* strukturální diverzitu, ale 92 % návrhů jsou jen HP substituce. Pokud edit distance počítá i hodnoty HP, měříte jitter, ne strukturu.

**Doporučení — tři úrovně diverzity:**
1. **Strukturní**: počet / entropie různých trojic (pre, fs, est) v populaci.
2. **Genotypová**: edit distance na celém genotypu včetně HP.
3. **Embeddingová**: kosinová vzdálenost v SNCS-D0 prostoru.

Poznámka k embeddingu: off-the-shelf sentence encoder na textu pipeline zachytí hlavně názvy estimátorů. „Embedding collapse při vysoké genotypové diverzitě“ je zajímavý výsledek, ale potřebuje vysvětlení (embedding ignoruje HP hodnoty?), ne jen konstatování.

### 3.7 Jediný uzavřený model
Haiku-class jako hlavní model a „reduced-scale Sonnet“ jako ablace je nejpravděpodobnější důvod zamítnutí. „Reduced-scale“ je navíc vágní.

**Doporučení:**
- Definovat ablaci modelu jako **všech 15 úloh, méně seedů** (např. 2), ne méně úloh.
- Zvážit **jeden open-weight model** (Qwen/Llama přes existující cross-vendor infrastrukturu z P4) místo nebo vedle Sonnetu. Řeší reprodukovatelnost (frozen uzavřený model se za rok změní) a EvoStar komunita to ocení víc než druhý model téhož vendora.
- H5 (kontrola, n = 6 úloh): Wilcoxon zde nemá sílu. Prezentovat jako mechanismovou plausibility check, ne jako test, a říct to dopředu.

### 3.8 Pozicování vůči LMX
Osa 2 („LLM pro mutaci i crossover v populaci“) je přesně to, co dělá Language Model Crossover (Meyerson et al.). Diferencovat explicitně: gramaticky vázaný prostor + kvantifikace chování operátoru, ne crossover samotný.

### 3.9 Drobnosti
- §4.9 a infrastrukturní bugy z §7 patří do supplementu / repozitáře, ne do 12 stran LNCS. V článku max. dvě věty o cost ledgeru a hard capu.
- „Independent of any method's outcome“ u výběru úloh je nadsazené — historická best-achievable accuracy pochází z běhů jiných metod na OpenML. Formulovat jako „nezávislé na výsledcích metod v tomto článku“.
- Row-capping velkých datasetů uvést explicitně s hodnotou.
- Klasické AutoML baseline: pro core claim nejsou nutné. Buď jen TPOT (evoluční, přirozený vztah) jako kalibrace s jasnou výhradou k neporovnatelnému prostoru/budgetu, nebo vynechat. Polovičaté zapojení GAMA před deadlinem přinese víc rizika než užitku.

---

## 4. Odpovědi na otázky pro recenzenta (§10 plánu)

1. **Anytime-at-common-budget vs. přesně rovné evaluace:** přesně rovné evaluace jako primární, anytime křivky různé délky jako sekundární.
2. **hp-decisive kontrola:** rozumný argument, ne statistický (n = 6). Prezentovat deskriptivně.
3. **Reduced-scale Sonnet:** nestačí. Plných 15 úloh s méně seedy; ideálně open-weight model.
4. **Accuracy jako fitness:** ano, s balanced accuracy sekundárně je to pro CC-18 v pořádku.
5. **TPOT/GAMA:** pro core claim nenutné; jen TPOT jako kalibrace, nebo nic.
6. **Kritická/mechanismová framing:** sedí na EvoLearn, za podmínky bodů 3.1–3.3. Bez nich to zní jako „slabý model s jedním promptem prohrál v malém prostoru“.

---

## 5. Návrh druhého (většího) prostoru — sekundární experiment

### 5.1 Proč nevyměnit primární prostor
- Pilot, opravený prompt, validity čísla i preregistrace jsou vázané na současnou gramatiku. Nový primární prostor = nový pilot, nový prompt, nové worked examples; dva měsíce do deadlinu.
- „Prostor vyčerpávajícím způsobem popsatelný v promptu“ je součást pozice: právě tam by LLM *mělo* mít výhodu. Že ji nemá ani tam, je silnější tvrzení než remíza v prostoru, který v promptu popsat nejde.
- Ve větším prostoru se objeví nové confoundy (delší prompt, více invalidů, hlubší search), které rozmažou mechanismovou část.

Jediný scénář pro výměnu primárního prostoru: pokud věříte, že ve větším prostoru LLM vyhraje. Pak by negativní paper stál na prostoru, o kterém sami víte, že je nevýhodný — to je „rigged“ námitka. Pokud tomu nevěříte, dvouúrovňový design je bezpečnější a informativnější.

### 5.2 Princip konstrukce: rozšířit, ne vyměnit
Zachovat formát genotypu `{stage: {name, params}}`, YAML gramatiku, prompt schéma, operátor interface, náhodný operátor i harness. Změnit **pouze gramatický blok** (a tím prompt prefix). Tím:
- H3 (validita) se ve větším prostoru testuje zdarma jako bonus;
- všechny metriky z §4.7 fungují beze změny;
- rozdíl mezi podmínkami je jen velikost prostoru, nic jiného.

Cíl: řádově **stovky až nízké tisíce struktur** místo 63, aby novelty nebyla nasycená po pár generacích, ale prostor byl stále popsatelný v promptu.

### 5.3 Navržená gramatika G2

| Stage | G1 (současná) | G2 (rozšířená) |
|---|---|---|
| preprocessing | none, standardize, minmax | none, standardize, minmax, robust, quantile(uniform/normal) |
| feature_engineering (**nový, volitelný**) | — | none, polynomial(degree ∈ {2,3}, interaction_only), binning(n_bins, strategy), variance_threshold(t) |
| feature_selection | none, selectk(k), pca(n) | none, selectk(k), select_percentile(p), pca(n), mutual_info(k), model_based(l1_C) |
| estimator | rf, extra_trees, hist_gbrt, logistic, svc, knn, gaussian_nb | + mlp(hidden_sizes, alpha, lr_init), gradient_boosting(n_est, lr, depth), adaboost(n_est, lr), decision_tree(depth, min_leaf), ridge_classifier(alpha), sgd(loss, alpha, penalty) |

Počet struktur: 5 × 4 × 6 × 13 = **1 560** (vs. 63). Se spojitými a kategoriálními HP je prostor řádově větší i na úrovni genotypů.

Poznámky k výběru:
- Přidané komponenty jsou standardní sklearn (žádný nový kód v harnessu kromě kompilace nových jmen).
- `polynomial` a `binning` zavádějí **interakce mezi stagi** (polynomial degree 3 + svc je drahé; binning + hist_gbrt je zbytečné) — dává prostoru netriviální strukturu, kde by LLM priors *měly* pomoci. To je záměr: G2 je prostor, kde má LLM nejlepší šanci vyhrát.
- `mlp` a `gradient_boosting` mají HP s výraznou interakcí (lr × n_est × depth) — hp-decisive část prostoru roste.
- Volitelnost stage feature_engineering (`none` jako default) drží kompatibilitu: každý G1 genotyp je validní G2 genotyp. To umožňuje **přímé srovnání G1 a G2 na stejných úlohách** — G1 je podprostor G2.

Bezpečnostní opatření v harnessu:
- **Timeout na evaluaci** (např. 5× medián G1 evaluace) — polynomial degree 3 + svc na širokých datech je jinak nepředvídatelné. Timeout = fitness 0 s error stringem, stejně jako pád.
- Row-cap pro G2 může být nutné snížit; zapsat do konfigu.
- `polynomial` omezit na datasety s ≤ ~50 rysy (v gramatice jako podmínka, nebo přijmout timeout).

### 5.4 Prompt pro G2
- Gramatický blok v prefixu roste přibližně 3–4×. Odhad: G1 prefix je pod 4096 tokenů (cache vypnuta); G2 prefix se pravděpodobně dostane **nad** 4096 tokenů → prompt caching na Haiku se zapne. Cenově pomůže, ale je to **rozdíl mezi podmínkami** — uvést v článku, nemá vliv na chování modelu.
- Worked examples: stejný počet jako v G1, ale pokrývající i nové stage. Struktury z worked examples evidovat a v novelty metrikách reportovat zvlášť (viz §3.2).
- Výstupní schéma nezměněno (jeden JSON genotyp).

### 5.5 Experimentální design G2
- **Úlohy:** 5 strukturně rozhodných úloh z preregistrovaných 15 (výběr předem: ty s největším headroomem, aby byl větší prostor využitelný).
- **Seedy:** 3.
- **Metody:** LLM-GA vs. random-operator GA (random search volitelně, pokud zbude budget). Klasické AutoML ne.
- **Budget:** stejný počet skutečných evaluací jako v G1 běhu na téže úloze — umožňuje srovnání G1 vs. G2 pro každou metodu zvlášť.
- **Populace/generace:** stejné jako G1. Alternativně (a lépe, pokud budget dovolí): 1.5× generací, protože větší prostor potřebuje déle na konvergenci; pak ale reportovat oba budgety.
- **Model:** Haiku (stejný jako hlavní běh). Sonnet/open-weight ablace na G2 pouze pokud primární G2 výsledek ukáže signál.
- Celkem: 5 úloh × 3 seedy × 2 metody = **30 běhů**.

### 5.6 Hypotéza a metriky
**H7 (velikost prostoru).** Rozdíl LLM-GA vs. random-GA v anytime best accuracy se při přechodu z G1 na G2 nezvětší ve prospěch LLM o více než mez ekvivalence z H1. (Preregistrovat; ať vyjde jakkoli, je to výsledek.)

Reportovat:
- Δ(LLM − random) na G1 vs. G2 pro stejných 5 úloh, s bootstrap CI.
- Strukturální novelty a pokrytí prostoru **po generacích** na G1 vs. G2 — v G2 by nasycení mělo nastat později; pokud LLM zůstane u ~8 % nových struktur i v G2, narrowness je vlastnost operátoru, ne artefakt malého prostoru.
- Validita (H3) v G2: očekávání, že roste podíl format errors u nových stagí — sám o sobě reportovatelný výsledek.
- Interakční chyby: podíl návrhů, které jsou gramaticky validní, ale evaluace selže/timeoutuje (polynomial+svc apod.). Toto je metrika, kde by LLM priors *měly* pomoci proti náhodnému operátoru — pokud nepomůžou, je to silný argument.

### 5.7 Odhad ceny
Base run G1: ~132 $ za 15 úloh × 5 seedů × 3 metody (LLM volání jen u LLM-GA) → řádově 1.8 $ na LLM-GA běh. G2: 15 LLM-GA běhů, prompt ~2–3× delší, ale s cachem → odhad **50–90 $** batch. S 1.5× generací ~80–130 $. Zapadá do rezervy programu (~585 $ → ~700 $).

### 5.8 Kde v článku
- Jedna podsekce v Results („Does a larger space change the picture?“), jedna figura (anytime Δ na G1 vs. G2 + novelty po generacích), gramatika G2 v supplementu.
- V Limitations přepsat „bounded space“ na „two space sizes; results may still not transfer to open-ended spaces“ — mnohem obhajitelnější.

---

## 6. Shrnutí změn do preregistrace před Phase 3

1. H1: přidat mez ekvivalence a TOST / CI-bound formulaci.
2. H4: přeformulovat nebo vypustit; vysvětlit a případně opravit náhodný operátor.
3. H5: označit jako deskriptivní kontrolu.
4. Přidat H7 (velikost prostoru) a design G2.
5. Ablace: přidat prompt bez worked examples / s výzvou k novým strukturám; teplotu; definovat model ablaci jako 15 úloh × 2 seedy; zvážit open-weight model.
6. Budget: primární design = přesně rovné skutečné evaluace; reportovat osu návrhů sekundárně.
7. Metriky: tři úrovně diverzity; novelty a pokrytí po generacích; evidence struktur z worked examples.
