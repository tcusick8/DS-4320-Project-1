# DS-4320-Project-1: NYC EMS Limited Resources and Unlimited Calls

> [!NOTE]
> This project develops a machine learning pipeline to improve NYC EMS resource allocation by predicting, at the moment a 911 call is received, whether a patient will require hospital transport — a proxy for call severity. Using over 14 million historical CAD incidents, a Random Forest classifier trained on intake-time features (call type, severity code, borough, time of day) achieves a ROC-AUC of 0.707, enabling real-time queue ranking that surfaces the most critical calls to dispatchers first without replacing any existing workflow or infrastructure.

---

| Spec | Value |
|---|---|
| Name | Thomas Cusick |
| NetID | tpg6hu |
| DOI | [https://doi.org/10.5281/zenodo.19228345](https://doi.org/10.5281/zenodo.19265212) |
| Press Release | [When Every Second Counts, AI Now Decides Who Gets Help First](Press_Release.md) |
| Data | [OneDrive Data Folder](https://1drv.ms/f/c/9e42f755abca0340/IgCP8jmuoy_UQZYjd_i5tb7pAepNvVgFgtUZ0LIixIIMgXQ?e=drhuHn) |
| Pipeline | [Pipeline](https://github.com/tcusick8/DS-4320-Project-1/tree/main/Pipeline) |
| License | [MIT](LICENSE.md) |

--- 

## Problem Definition

**General Problem:**

Allocating emergency response resources

**Specific refined statement:**

Using the NYC EMS Incident Dispatch Data (NYC Open Data), this project develops a real-time call queue ranking model that, given a set of simultaneously active, unassigned 911 EMS calls, scores and orders each call by predicted clinical urgency, using only information available at the moment of intake (call type, reported symptoms, borough, time of day, and historical severity patterns for that call category), so that the most life-threatening incidents are consistently dispatched first and dangerous wait times for Priority 1 calls are minimized.

**Rationale for refinement:**

The general problem of emergency resource allocation is vast — it encompasses staffing schedules, ambulance positioning, hospital routing, and supply chain logistics. To make the project tractable, the refined statement focuses on a single, high-stakes decision point: the moment when multiple 911 calls are queued simultaneously and a dispatcher must decide which one gets an ambulance first. This framing was chosen because it is where human judgment is most fallible and where a data-driven model can deliver the clearest marginal improvement. When calls arrive one at a time, experienced dispatchers perform well; it is the concurrent surge — ten calls in ninety seconds during a Friday night in the Bronx — where misallocation is most likely and most costly. The NYC EMS Incident Dispatch Data supports this refinement directly: it contains call type codes, final incident classifications, assigned priority levels, and precise timestamps that allow reconstruction of simultaneous call queues. The target variable, whether a call results in hospital transport, is already encoded in `FIRST_TO_HOSP_DATETIME`, giving a clean supervised learning label. Restricting model inputs to intake-time features only enforces the real-world constraint that the model must make its ranking decision before any ambulance is en route, preventing data leakage and ensuring the model is deployable in production.

**Motivation:**

In emergency medicine, time is the most critical resource. For cardiac arrest, the chance of survival decreases by roughly 10% for every minute without defibrillation. For stroke, every 15-minute delay in treatment costs the average patient nearly a month of healthy life. Yet across the United States, EMS systems are under mounting pressure: call volumes have risen sharply over the past decade, ambulance staffing shortages have worsened since the COVID-19 pandemic, and urban 911 centers routinely face moments where demand exceeds available units. New York City alone processes over 1.6 million EMS calls per year — more than 4,000 per day — and a significant fraction arrive in simultaneous clusters during peak hours, major events, and emergencies. The status quo response is a human dispatcher applying a memorized priority code matrix — a system designed for a different era of call volume that has not been systematically modernized for the age of machine learning. An AI-powered queue ranking model represents a low-barrier, high-impact intervention: it does not replace dispatchers, it augments them, surfacing a ranked list that a human can accept, override, or query in real time.

[When Every Second Counts, AI Now Decides Who Gets Help First](Press_Release.md)

---

## Domain Exposition

**Terminology:**

| Term / KPI | Definition |
|---|---|
| **Priority 1 (P1)** | The highest EMS urgency level — immediate life threat, dispatched with lights and sirens. Examples: cardiac arrest, stroke, active shooting. |
| **Priority 2 (P2)** | Serious but not immediately life-threatening; expedited response without lights and sirens. |
| **Priority 3 (P3)** | Non-urgent calls; scheduled or routine response. |
| **CAD** | Computer-Aided Dispatch — the software system that receives 911 calls, logs incident data, and assigns units. |
| **Call Type Code** | A standardized alphanumeric code assigned at intake describing the nature of the reported emergency (e.g., `UNCONSCIOUS`, `CHEST PAIN`). |
| **Final Incident Type** | The post-dispatch classification of what the emergency actually was, assigned by the responding unit — used as the ground truth label. |
| **Dispatch Time** | Elapsed seconds from call receipt to unit assignment. A key EMS performance KPI. |
| **Response Time** | Elapsed seconds from call receipt to ambulance arrival on scene. The primary clinical outcome metric. |
| **Unit Hour Utilization (UHU)** | Proportion of time an ambulance unit spends on active calls. High UHU signals system saturation. |
| **MPDS** | Medical Priority Dispatch System — the most widely used structured protocol for prioritizing EMS calls at intake. Our model supplements this. |
| **Learning to Rank (LTR)** | ML techniques (e.g., RankNet, LambdaMART) designed to produce an ordered list rather than a classification or regression output. |
| **NDCG** | Normalized Discounted Cumulative Gain — a standard ranking quality metric measuring how well the model places the most urgent calls at the top. |
| **Triage** | The process of sorting calls by severity to allocate limited resources to those with greatest need. |
| **Borough** | One of NYC's five administrative divisions (Manhattan, Brooklyn, Queens, The Bronx, Staten Island) — a key geographic feature. |

**Domain:**

This project sits at the intersection of **emergency medical services (EMS) operations research**, **public safety technology**, and **applied machine learning**. The broader domain is often called *emergency dispatch optimization* or *intelligent public safety systems*, and it draws from clinical medicine (understanding what makes a call life-threatening), operations research (queue theory, resource scheduling), and computer science (real-time decision support, learning-to-rank algorithms). EMS dispatch has historically been governed by rule-based protocols — most prominently MPDS — which assign priority codes through a structured decision tree of dispatcher questions. While these protocols provide consistency, they were not designed to handle the simultaneous queue problem: when ten calls arrive at once and only two ambulances are available, MPDS assigns each call a code independently but offers no mechanism for ranking them relative to each other under scarcity. This is the gap that machine learning is uniquely positioned to fill.

> [!TIP]
> Background reading materials are available in the [Background Reading Folder](https://drive.google.com/drive/folders/1aN9kYJriacFxD4SpTq3pUA4O3I13SS7t?usp=sharing)

| # | Title | Brief Description | Link |
|---|---|---|---|
| 1 | NYC EMS Incident Dispatch Data | The primary dataset. Contains millions of EMS dispatch records with call type, priority level, timestamps, borough, and unit response data. | [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj) |
| 2 | Forecasting Emergency Medical Service Call Arrival Rates (Matteson et al., 2011) | Foundational ML paper on EMS call volume forecasting using dynamic factor models. Establishes the temporal and spatial patterns in call demand relevant to understanding surge windows. | [Annals of Applied Statistics](https://arxiv.org/pdf/1107.4919) |
| 3 | Emergency Medical Services Dispatch Priority Prediction Using ML (Blomberg et al., 2019) | Applies ML to Danish EMS data to predict Priority 1 response at intake. The closest existing work to this project; provides modeling baselines and feature engineering strategies. | [PLOS ONE](https://doi.org/10.1371/journal.pone.0225135) |
| 4 | Learning to Rank for Information Retrieval (Liu, 2009) | Canonical tutorial on learning-to-rank methods (pointwise, pairwise, listwise) — essential technical background for the ranking model architecture. | [ACM Digital Library](https://dl.acm.org/doi/10.1561/1500000016) |
| 5 | Racial and Ethnic Disparities in EMS Response to Cardiac Arrest (Schieb et al., 2023) | Documents how response time disparities across demographic groups worsen outcomes — the equity context that makes correct call prioritization not just efficient, but just. | [NEJM](https://www.nejm.org/doi/full/10.1056/NEJMoa2200798) |

---

## Data Creation

**Raw Data Acquisition:**

The raw data was obtained from NYC Open Data, the City of New York's official public data portal, at the following URL: https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj. The dataset is published and maintained by the New York City Fire Department (FDNY) EMS Command and is updated regularly. It contains one record per EMS incident dispatch, covering calls received by the NYC 911 system from 2005 through early 2026 — 31 columns and approximately 8 GB in size.

Because the full file exceeds the memory and storage constraints of a standard analysis environment, two processing steps were applied after download. First, the file was read in 100,000-row chunks and a random 50% sample was drawn from each chunk, producing a reduced file that preserves the full temporal and geographic range of the original data. Second, the 31 original columns were filtered to the 14 directly relevant to the queue-ranking model, retaining only features available at call intake time (to prevent data leakage) plus key outcome and operational variables. No records were fabricated or imputed at this stage; the file is a direct subset of the original NYC Open Data release.

**Code:**

| File | Description | Link |
|------|-------------|------|
| `EMS_data_creation.ipynb` | Chunks and samples the raw 8GB CSV, filters to 14 columns, exports `EMS.csv` | [EMS_data_creation.ipynb](Pipeline/EMS_data_creation.ipynb) |
| `EMS_RandomForest.ipynb` | Loads parquet tables via DuckDB, engineers features, trains Random Forest classifier, outputs evaluation metrics and visualizations | [EMS_RandomForest.ipynb](Pipeline/EMS_RandomForest.ipynb) |

> [!WARNING]
> Four meaningful sources of bias are present in this dataset and should be understood before modeling.

**Bias Identification:**

| Bias Type | Description |
|---|---|
| **Reporting bias** | Only incidents that prompted a 911 call are included. Emergencies resolved by bystanders or via private transport are entirely absent, skewing the data toward higher-acuity events. |
| **Geographic/socioeconomic bias** | Call volume, response time, and incident type distributions vary across boroughs in ways that reflect underlying inequality. Models could inadvertently encode neighborhood as a proxy for resource allocation. |
| **Temporal bias** | The dataset spans 2005–2026, including major operational changes to NYC EMS and the COVID-19 pandemic (2020–2022), which dramatically distorted call volume and type distributions. |
| **Label bias** | `FIRST_TO_HOSP_DATETIME` is determined by the responding unit on scene, meaning it reflects both true clinical severity and individual responder judgment, introducing inter-rater variability into the ground truth. |

**Bias Mitigation:**

| Bias Type | Mitigation |
|---|---|
| **Reporting bias** | Acknowledged as an irreducible structural limitation. Conclusions are scoped to dispatch prioritization given that a call was made, not population-level emergency incidence. |
| **Geographic bias** | Performance metrics computed separately per borough. If the model systematically underperforms in lower-income areas, fairness-aware re-ranking or training-example re-weighting will be applied. |
| **Temporal bias** | Chronological train/test splitting — training on incidents before a cutoff date, evaluating on incidents after — prevents temporal leakage and ensures the model is tested on data resembling its deployment context. |
| **Label bias** | Acknowledged as irreducible noise in the ground truth. Flagged as a key source of uncertainty; all evaluation metrics reported with confidence intervals. |

**Rationale:**

Three critical decisions shaped the dataset used in this project. First, the raw 8 GB file was reduced via 50% random chunk sampling rather than date-range filtering. This preserves the full 2005–2026 temporal range, all five boroughs, and all call-type categories — breadth essential for capturing seasonal patterns and long-term trends. The tradeoff is potential underrepresentation of rare call types and geographic combinations, which is addressed through class weighting during modeling. Second, 14 of 31 columns were retained. The 17 dropped columns fall into two categories: redundant administrative geographies (police precinct, city council district) that add no predictive signal beyond `BOROUGH` and `ZIPCODE`, and post-dispatch operational timestamps generated after the dispatch decision is made that cannot be used as model inputs without introducing data leakage. Third, `FIRST_TO_HOSP_DATETIME` was chosen as the basis for the target variable rather than `FINAL_SEVERITY_LEVEL_CODE`. Hospital transport is a concrete, objectively recorded outcome — a unit either drove to a hospital or it did not — whereas severity codes involve responder judgment and inter-rater variability. The binary transport label gives a cleaner supervised learning signal while remaining a clinically meaningful proxy for call urgency.

---

## Metadata

**Schema**

**EMS.csv**
| Key | Field | Type |
|-----|-------|------|
| PK | CAD_INCIDENT_ID | BIGINT |
| | INCIDENT_DATETIME | VARCHAR |
| | INITIAL_CALL_TYPE | VARCHAR |
| | INITIAL_SEVERITY_LEVEL_CODE | FLOAT |
| | FINAL_CALL_TYPE | VARCHAR |
| | FINAL_SEVERITY_LEVEL_CODE ← TARGET | FLOAT |
| | VALID_DISPATCH_RSPNS_TIME_INDC | VARCHAR |
| | DISPATCH_RESPONSE_SECONDS_QY | INTEGER |
| | INCIDENT_RESPONSE_SECONDS_QY | INTEGER |
| | HELD_INDICATOR | VARCHAR |
| | INCIDENT_DISPOSITION_CODE | VARCHAR |
| | BOROUGH | VARCHAR |
| | ZIPCODE | FLOAT |
| | FIRST_TO_HOSP_DATETIME | BINARY |


**Data Tables**

| Table | Description | Rows | File |
|-------|-------------|------|------|
| INCIDENTS | Core incident records including timestamps, hold status, and disposition codes. | 14,348,689 | [incidents.parquet](https://1drv.ms/u/c/9e42f755abca0340/IQDXCDac8uugQYhc5KfV7qrqAY3V8F4zAe0tt3c7NCNcfOE?e=Ufnr0t) |
| SEVERITY | Call type classification (initial and final) and severity level codes. | 14,348,689 | [severity.parquet](https://1drv.ms/u/c/9e42f755abca0340/IQAcp202kmH2SK9jUtKei4bqATDuvCnfd6wISuBoG3_iWDI?e=6Tl8F4) |
| DISPATCH | Dispatch and incident response times, hospital transport timestamp, and validity flag. | 14,348,689 | [dispatch.parquet](https://1drv.ms/u/c/9e42f755abca0340/IQC88RqbFDY-TZEYGY0MwbprAXk6knfKkC9AaNCX7my1LC8?e=BQM6Uy) |
| LOCATION | Geographic identifiers (borough and ZIP code) per incident. | 14,348,689 | [location.parquet](https://1drv.ms/u/c/9e42f755abca0340/IQCG23dItlsZRoBrtaNeb9nkAQ7nq5l1RFjmBCWWuK-pgBs?e=bU40A0) |

**Data Dictionary**

| Feature Name | Table | Data Type | Description | Example |
|---|---|---|---|---|
| `CAD_INCIDENT_ID` | ALL | Integer | Unique identifier for each CAD incident. Shared join key across all tables. | 101211759 |
| `INCIDENT_DATETIME` | INCIDENTS | Timestamp | Date and time the incident was recorded. | 2010-05-01 12:04:42 |
| `HELD_INDICATOR` | INCIDENTS | String (Y/N) | Whether the incident was held/queued before dispatch. | N |
| `INCIDENT_DISPOSITION_CODE` | INCIDENTS | Integer | Numeric code for the final outcome of the incident. ~18% null. | 93 |
| `INITIAL_CALL_TYPE` | SEVERITY | String | Emergency call category at initial dispatch. | INJURY |
| `INITIAL_SEVERITY_LEVEL_CODE` | SEVERITY | Integer | Severity level at initial classification. 1 = most severe. | 5 |
| `FINAL_CALL_TYPE` | SEVERITY | String | Call type category after incident resolution. | INJURY |
| `FINAL_SEVERITY_LEVEL_CODE` | SEVERITY | Integer | Severity level after incident resolution. 1 = most severe. | 5 |
| `VALID_DISPATCH_RSPNS_TIME_INDC` | DISPATCH | String (Y/N) | Whether the dispatch response time is a valid measurement. | Y |
| `DISPATCH_RESPONSE_SECONDS_QY` | DISPATCH | Integer | Seconds from incident creation to unit dispatch. | 25 |
| `INCIDENT_RESPONSE_SECONDS_QY` | DISPATCH | Integer | Seconds from incident creation to unit arrival on scene. ~3.8% null. | 322 |
| `FIRST_TO_HOSP_DATETIME` | DISPATCH | Timestamp | Date and time the first unit departed for hospital. Null if no transport occurred — used as target variable proxy for call severity. | 2010-05-01 13:12:04 |
| `BOROUGH` | LOCATION | String | NYC borough of the incident. | MANHATTAN |
| `ZIPCODE` | LOCATION | String | ZIP code of the incident. ~2% null. | 10002 |

**Data Quality & Uncertainty**

| Feature Name | Null Count | Null % | Notes |
|---|---|---|---|
| `FIRST_TO_HOSP_DATETIME` | ~4,574,891 | ~31.9% | Null indicates no hospital transport. Used as target variable — null = 0 (not transported), not null = 1 (transported). |
| `INCIDENT_DISPOSITION_CODE` | ~2,582,764 | ~18.0% | High null rate; may reflect unresolved or cancelled incidents. Exclude or impute with caution. |
| `INCIDENT_RESPONSE_SECONDS_QY` | 542,305 | ~3.8% | Likely incidents where units were never dispatched or arrival was not logged. Drop or right-censor for response-time analysis. |
| `ZIPCODE` | 292,268 | ~2.0% | Borough is still available. Missing ZIPs can bias ZIP-level aggregations. |
| `DISPATCH_RESPONSE_SECONDS_QY` | 0 | 0.0% | Fully populated. Reliable baseline for dispatch performance analysis. |
| `HELD_INDICATOR` | 0 | 0.0% | Fully populated. Binary flag safe to use as-is. |
| `VALID_DISPATCH_RSPNS_TIME_INDC` | 0 | 0.0% | Fully populated. Filter to `Y` for valid dispatch timing subsets. |
| `INITIAL_SEVERITY_LEVEL_CODE` | 0 | 0.0% | No nulls. Compare within call-type cohorts only — scale varies by type. |
| `FINAL_SEVERITY_LEVEL_CODE` | 0 | 0.0% | No nulls. Compare to initial code to detect severity escalations. |

