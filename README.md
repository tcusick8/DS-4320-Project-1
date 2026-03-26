# DS-4320-Project-1: NYC EMS Limited Resources and Unlimited Calls

### Executive Summary
Executive summary, short paragraph explainaing contents of repo in executive form

<br>

---

Name - Thomas Cusick

NetID - tpg6hu

DOI: - [https://doi.org/10.5281/zenodo.19228345](https://doi.org/10.5281/zenodo.19228345)

Press Release - [When Every Second Counts, AI Now Decides Who Gets Help First](Press_Release.md)

[OneDrive Data Folder](https://1drv.ms/f/c/9e42f755abca0340/IgCP8jmuoy_UQZYjd_i5tb7pAepNvVgFgtUZ0LIixIIMgXQ?e=drhuHn)

[Pipeline](https://github.com/tcusick8/DS-4320-Project-1/tree/main/Pipeline)

License?? - state name of license here and link to file in top level of repo (use normal github conventions)

---

## Problem Definition:
**General Problem:** 

Finite EMS resources for near infinite 911 calls

**Specific refined statement:** 

Using the NYC EMS Incident Dispatch Data (NYC Open Data), this project develops a real-time call queue ranking model that, given a set of simultaneously active, unassigned 911 EMS calls, scores and orders each call by predicted clinical urgency, using only information available at the moment of intake (call type, reported symptoms, borough, time of day, and historical severity patterns for that call category), so that the most life-threatening incidents are consistently dispatched first and dangerous wait times for Priority 1 calls are minimized.


**Rationale for refinement:** 

The general problem of emergency resource allocation is vast, it encompasses staffing schedules, ambulance positioning, hospital routing, and supply chain logistics. To make the project tractable, the refined statement focuses on a single, high-stakes decision point: the moment when multiple 911 calls are queued simultaneously and a dispatcher must decide which one gets an ambulance first. This framing was chosen because it is where human judgment is most fallible and where a data-driven model can deliver the clearest marginal improvement. When calls arrive one at a time, experienced dispatchers perform well; it is the concurrent surge, ten calls in ninety seconds during a Friday night in the Bronx, where misallocation is most likely and most costly. The NYC EMS Incident Dispatch Data supports this refinement directly: it contains call type codes, final incident classifications, assigned priority levels, and precise timestamps that allow reconstruction of simultaneous call queues. The target variable, whether a call escalates to or is confirmed as Priority 1 (immediate life threat), is already encoded in the dataset's final priority field, giving a clean supervised learning label. Restricting model inputs to intake-time features only (no post-dispatch information) enforces the real-world constraint that the model must make its ranking decision before any ambulance is en route, preventing data leakage and ensuring the model is deployable in production.


**Motivation:** 

In emergency medicine, time is the most critical resource. For cardiac arrest, the chance of survival decreases by roughly 10% for every minute without defibrillation. For stroke, every 15-minute delay in treatment costs the average patient nearly a month of healthy life. Yet across the United States, EMS systems are under mounting pressure: call volumes have risen sharply over the past decade, ambulance staffing shortages have worsened since the COVID-19 pandemic, and urban 911 centers routinely face moments where demand exceeds available units. New York City alone processes over 1.6 million EMS calls per year, more than 4,000 per day, and a significant fraction of those arrive in simultaneous clusters during peak hours, major events, and emergencies. The status quo response to this problem is a human dispatcher applying a memorized priority code matrix, a system that was designed for a different era of call volume and has not been systematically modernized for the age of machine learning. An AI-powered queue ranking model represents a low-barrier, high-impact intervention: it does not replace dispatchers, it augments them, surfacing a ranked list that a human can accept, override, or query in real time. Beyond saving individual lives, a more rational dispatch queue reduces inequities in response time across neighborhoods, reduces ambulance repositioning costs, and provides EMS leadership with a new instrument for measuring and improving dispatch quality at scale. This project lays the analytical groundwork for that system using one of the richest open EMS datasets in the world.


[When Every Second Counts, AI Now Decides Who Gets Help First](Press_Release.md)

---

## Domain Exposition


**Terminology:**

| Term / KPI | Definition |
|---|---|
| **Priority 1 (P1)** | The highest EMS urgency level — immediate life threat, dispatched with lights and sirens. Examples: cardiac arrest, stroke, active shooting. |
| **Priority 2 (P2)** | Serious but not immediately life-threatening; expedited response without lights and sirens. |
| **Priority 3 (P3)** | Non-urgent calls; scheduled or routine response. |
| **CAD (Computer-Aided Dispatch)** | The software system that receives 911 calls, logs incident data, and assigns units. The model would integrate with or augment CAD output. |
| **Call Type Code** | A standardized alphanumeric code assigned at intake describing the nature of the reported emergency (e.g., "UNCONSCIOUS," "CHEST PAIN"). |
| **Final Incident Type** | The post-dispatch classification of what the emergency actually was, assigned by the responding unit — used as the ground truth label. |
| **Dispatch Time** | The elapsed time in seconds from when a call is received to when a unit is assigned. A key performance KPI for EMS operations. |
| **Response Time** | The elapsed time from call receipt to ambulance arrival on scene. The primary clinical outcome metric. |
| **Unit Hour Utilization (UHU)** | The proportion of time an ambulance unit spends on active calls; a key operational efficiency KPI. High UHU signals system saturation. |
| **Concurrent Call Volume** | The number of active, unassigned 911 calls in the queue at a given moment — the core problem trigger for this project. |
| **Learning to Rank (LTR)** | A family of machine learning techniques (e.g., RankNet, LambdaMART) designed to produce an ordered list rather than a classification or regression output. |
| **MPDS (Medical Priority Dispatch System)** | The most widely used structured protocol for prioritizing EMS calls at intake; assigns a deterministic code based on dispatcher questions. Our model supplements this. |
| **Triage** | The clinical process of sorting patients or calls by severity to allocate limited resources to those with greatest need. |
| **Turnaround Time** | Total time an EMS unit is committed to a single call, from dispatch through return to available status. |
| **Borough** | One of NYC's five administrative divisions (Manhattan, Brooklyn, Queens, The Bronx, Staten Island) — a key geographic feature in the dataset. |
| **NDCG (Normalized Discounted Cumulative Gain)** | A standard ranking quality metric; measures how well the model places the most urgent calls at the top of the ranked list. |


**Domain:**

This project sits at the intersection of **emergency medical services (EMS) operations research**, **public safety technology**, and **applied machine learning**. The broader domain is often called *emergency dispatch optimization* or *intelligent public safety systems*, and it draws from clinical medicine (understanding what makes a call life-threatening), operations research (queue theory, resource scheduling), and computer science (real-time decision support, learning-to-rank algorithms). EMS dispatch has historically been governed by rule-based protocols — most prominently the Medical Priority Dispatch System (MPDS) — which assign priority codes through a structured decision tree of dispatcher questions. While these protocols provide consistency, they were not designed to handle the simultaneous queue problem: when ten calls arrive at once and only two ambulances are available, MPDS assigns each call a code independently but offers no mechanism for ranking them relative to each other under scarcity. This is the gap that machine learning is uniquely positioned to fill. NYC's open EMS dispatch dataset represents one of the most detailed public records of urban emergency response in existence, making it an ideal testbed for developing and validating a data-driven queue-ranking system that could be generalized to other major cities.


[Background Reading Folder](https://drive.google.com/drive/folders/1aN9kYJriacFxD4SpTq3pUA4O3I13SS7t?usp=sharing)


**Reading Summaries**

| # | Title | Brief Description | Link |
|---|---|---|---|
| 1 | **NYC EMS Incident Dispatch Data** (NYC Open Data) | The primary dataset. Contains millions of EMS dispatch records with call type, priority level, timestamps, borough, and unit response data for New York City. Updated regularly. | [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj) |
| 2 | **Forecasting Emergency Medical Service Call Arrival Rates** (Matteson et al., 2011) | Foundational statistical/ML paper on EMS call volume forecasting using dynamic factor models and time-series methods. Establishes the temporal and spatial patterns in call demand — directly relevant to understanding when simultaneous surge windows occur. | [Annals of Applied Statistics](https://arxiv.org/pdf/1107.4919) |
| 3 | **Emergency Medical Services Dispatch Priority Prediction Using Machine Learning** (Blomberg et al., 2019) | Applies ML to Danish EMS data to predict whether a call requires a Priority 1 response at intake. The closest existing work to our problem; provides modeling baselines and feature engineering strategies. | [PLOS ONE](https://doi.org/10.1371/journal.pone.0225135) |
| 4 | **Learning to Rank for Information Retrieval** (Liu, 2009) | The canonical textbook chapter/tutorial on learning-to-rank methods (pointwise, pairwise, listwise). Essential technical background for the ranking model architecture this project will use. | [ACM Digital Library](https://dl.acm.org/doi/10.1561/1500000016) |
| 5 | **Racial and Ethnic Disparities in Emergency Medical Services Response to Cardiac Arrest** (Schieb et al., 2023) | Documents how EMS response time disparities across demographic groups lead to worse outcomes for minority communities — the equity context that makes correct call prioritization not just efficient, but just. | [The New England Journal of Medicine](https://www.nejm.org/doi/full/10.1056/NEJMoa2200798) |


---

## Data Creation

**Raw Data Acquisition:**
The raw data was obtained from NYC Open Data, the City of New York's official public data portal, at the following URL: https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj. The dataset is published and maintained by the New York City Fire Department (FDNY) EMS Command and is updated regularly. It contains one record per EMS incident dispatch, covering calls received by the NYC 911 system from 2005 through early 2026. The full dataset was downloaded as a CSV file, approximately 8 GB in size, containing 31 columns and several million records spanning over two decades of EMS operations.

Because the full 8 GB file exceeds the memory and storage constraints of a standard analysis environment, two processing steps were applied after download. First, the file was read in 100,000-row chunks, and a random 18.75% sample was drawn from each chunk, producing a reduced file of approximately 1.5 GB that preserves the full temporal and geographic range of the original data. Second, the 31 original columns were filtered down to the 13 columns directly relevant to the queue-ranking model, retaining only features available at call intake time (to prevent data leakage) plus key outcome and operational variables. No records were fabricated or imputed at this stage; the file is a direct subset of the original NYC Open Data release.


**Code**
```python
import pandas as pd

sampled_df = []
chunksize = 100000 
for chunk in pd.read_csv('OG_8GB_EMS_Incident_Dispatch_Data_20260315.csv', chunksize=chunksize):
    sampled_df.append(chunk.sample(frac=0.50))

EMS_31 = pd.concat(sampled_df)

keep_cols = [
    'CAD_INCIDENT_ID',
    'INCIDENT_DATETIME',
    'INITIAL_CALL_TYPE',
    'INITIAL_SEVERITY_LEVEL_CODE',
    'FINAL_CALL_TYPE',
    'FINAL_SEVERITY_LEVEL_CODE',
    'VALID_DISPATCH_RSPNS_TIME_INDC',
    'DISPATCH_RESPONSE_SECONDS_QY',
    'INCIDENT_RESPONSE_SECONDS_QY',
    'HELD_INDICATOR',
    'INCIDENT_DISPOSITION_CODE',
    'BOROUGH',
    'ZIPCODE',
]

EMS = EMS_31[keep_cols]
EMS.to_csv('EMS.csv', index=False)
```


**Bias Identification**
The dataset contains several meaningful sources of bias rooted in the data collection process. First, there is reporting bias: the dataset only includes incidents that prompted a 911 call. Emergencies resolved by bystanders, handled via private transport, or never recognized as emergencies are entirely absent, skewing the data toward higher-acuity events and likely underrepresenting lower-severity incidents with clinical significance. Second, geographic and socioeconomic bias is present because call volume, response time, and incident type distributions vary across boroughs and zip codes in ways that reflect underlying inequality. Neighborhoods with higher poverty rates and greater population density exhibit systematically different call patterns that a model could learn and inadvertently encode as a legitimate signal. Third, temporal bias exists because the dataset spans 2005–2026, a period that includes major operational changes to NYC EMS, evolving classification protocols, and the COVID-19 pandemic, which dramatically distorted call volume and type distributions from 2020–2022. Patterns from earlier years may not generalize to current conditions. Finally, label bias affects the target variable itself: FINAL_SEVERITY_LEVEL_CODE is assigned by the responding unit upon arrival on scene, meaning it reflects both true clinical severity and individual responder judgment, thereby introducing inter-rater variability into the ground truth.


**Bias Mitigation**
Each identified bias is addressed or bounded as follows. Reporting bias is an irreducible structural limitation of 911-sourced data; it is explicitly acknowledged, and conclusions are scoped accordingly — the model describes dispatch prioritization given that a call was made, not population-level emergency incidence. Geographic bias is mitigated by computing performance metrics (Normalized Discounted Cumulative Gain (NDCG) or precision at k) separately for each borough and for zip codes stratified by median income; if the model systematically underperforms in lower-income areas, fairness-aware re-ranking or training-example re-weighting will be applied. Temporal bias is addressed through chronological train/test splitting — training on incidents before a cutoff date and evaluating on incidents after the cutoff — which prevents temporal leakage and ensures the model is tested on data that resembles its deployment context. Label bias in severity codes is acknowledged as irreducible noise in the ground truth; it is flagged as a key source of uncertainty rather than corrected, since no cleaner labels exist in the dataset. Sampling bias introduced by the chunked random sample is quantified using bootstrap resampling, and all model evaluation metrics are reported with confidence intervals.


**Rationale**
Three critical decisions shaped the dataset used in this project. First, the raw 8 GB file was reduced to approximately 1.5 GB via 18.75% random chunk sampling. This was chosen over date-range filtering because it preserves the full 2005–2026 temporal range, all five boroughs, and all call-type categories — a breadth essential for capturing seasonal patterns and long-term trends. The tradeoff is potential underrepresentation of rare call types and geographic combinations, which is documented above and addressed through class weighting during modeling. Second, 13 of 31 columns were retained. The 18 dropped columns fall into two categories: redundant administrative geographies (police precinct, city council district, etc.) that add no predictive signal beyond BOROUGH and ZIPCODE, and post-dispatch operational timestamps that are generated after the dispatch decision is made and therefore cannot be used as model inputs without introducing data leakage. Retaining them would inflate training accuracy but render the model non-functional for real-time deployment. HELD_INDICATOR was specifically kept because it directly flags calls that entered a queue — the core scenario the model is designed to address. Third, FINAL_SEVERITY_LEVEL_CODE was chosen as the target variable rather than INITIAL_SEVERITY_LEVEL_CODE. The initial code reflects the dispatcher's intake assessment — precisely what the model aims to improve upon — so using it as the label would conflate input with output. The final code represents the post-dispatch ground-truth assessment of actual clinical severity, making it the most appropriate available proxy for true urgency despite the variability in responder judgment noted above.

## Metadata

**Schema**

fhfhf


**Data**

| Table | Description | Rows | CSV File |
|-------|-------------|------|----------|
| INCIDENTS | Core incident records including timestamps, hold status, and disposition codes. | 14,348,689 | [incidents.csv](https://1drv.ms/x/c/9e42f755abca0340/IQBg8p8YiKviQICcbIpYdJ77ATn96-Q5JFByQjimI0YeHpU?e=zpUWL4) |
| SEVERITY | Call type classification (initial and final) and severity level codes. | 14,348,689 | [severity.csv](https://1drv.ms/x/c/9e42f755abca0340/IQCvcjFDObwgT4E_MaYYL87tARP4CDzAU_kTDYEPdoq1fWc?e=FF3v4N) |
| DISPATCH | Dispatch and incident response times, plus a validity flag. | 14,348,689 | [dispatch.csv](https://1drv.ms/x/c/9e42f755abca0340/IQBWqWtRl0AXSIBGAc1ezCGtAdoXdrmDGLwEmE4xh7TTEo8?e=LwDswn) |
| LOCATION | Geographic identifiers (borough and ZIP code) per incident. | 14,348,689 | [location.csv](https://1drv.ms/x/c/9e42f755abca0340/IQAOeMyUFuDdR4rhi2HixqXIATITJ98mMgOPuRj95cXtXzE?e=wcX4q3) |

---

**Data Dictionary** (metadata)

| Feature Name | Table | Data Type | Description | Example |
|---|---|---|---|---|
| CAD_INCIDENT_ID | ALL | Integer | Unique identifier for each CAD incident. Shared join key across all tables. | 101211759 |
| INCIDENT_DATETIME | INCIDENTS | Timestamp | Date and time the incident was recorded. | 2010-05-01 12:04:42 |
| HELD_INDICATOR | INCIDENTS | String (Y/N) | Whether the incident was held/queued before dispatch. | N |
| INCIDENT_DISPOSITION_CODE | INCIDENTS | Integer | Numeric code for the final outcome of the incident. ~18% null. | 93 |
| INITIAL_CALL_TYPE | SEVERITY | String | Emergency call category at initial dispatch. | INJURY |
| INITIAL_SEVERITY_LEVEL_CODE | SEVERITY | Integer | Severity level at initial classification. | 5 |
| FINAL_CALL_TYPE | SEVERITY | String | Call type category after incident resolution. | INJURY |
| FINAL_SEVERITY_LEVEL_CODE | SEVERITY | Integer | Severity level after incident resolution. | 5 |
| VALID_DISPATCH_RSPNS_TIME_INDC | DISPATCH | String (Y/N) | Whether the dispatch response time is a valid measurement. | Y |
| DISPATCH_RESPONSE_SECONDS_QY | DISPATCH | Integer | Seconds from incident creation to unit dispatch. | 25 |
| INCIDENT_RESPONSE_SECONDS_QY | DISPATCH | Integer | Seconds from incident creation to unit arrival on scene. ~3.8% null. | 322 |
| BOROUGH | LOCATION | String | NYC borough of the incident. | MANHATTAN |
| ZIPCODE | LOCATION | String | ZIP code of the incident. ~2% null. | 10002 |

---

**Data Dictionary** (uncertainty)

| Feature Name | Null Count | Null % | Notes |
|---|---|---|---|
| INCIDENT_DISPOSITION_CODE | ~2,582,764 | ~18.0% | High null rate; may reflect unresolved or cancelled incidents. Exclude or impute with caution. |
| INCIDENT_RESPONSE_SECONDS_QY | 542,305 | ~3.8% | Likely incidents where units were never dispatched or arrival was not logged. Drop or right-censor for response-time analysis. |
| ZIPCODE | 292,268 | ~2.0% | Borough is still available. Missing ZIPs can bias ZIP-level aggregations. |
| DISPATCH_RESPONSE_SECONDS_QY | 0 | 0.0% | Fully populated. Reliable baseline for dispatch performance analysis. |
| HELD_INDICATOR | 0 | 0.0% | Fully populated. Binary flag safe to use as-is. |
| VALID_DISPATCH_RSPNS_TIME_INDC | 0 | 0.0% | Fully populated. Filter to Y for valid dispatch timing subsets. |
| INITIAL_SEVERITY_LEVEL_CODE | 0 | 0.0% | No nulls. Compare within call-type cohorts only — scale varies by type. |
| FINAL_SEVERITY_LEVEL_CODE | 0 | 0.0% | No nulls. Compare to initial code to detect severity escalations. |
























