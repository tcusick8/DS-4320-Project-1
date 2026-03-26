# DS-4320-Project-1: NYC EMS Limited Resources and Unlimited Calls

### Executive Summary
Executive summary, short paragraph explainaing contents of repo in executive form

<br>

---

Name - Thomas Cusick

NetID - tpg6hu

DOI: - [https://doi.org/10.5281/zenodo.19228345](https://doi.org/10.5281/zenodo.19228345)

Press Release - [When Every Second Counts, AI Now Decides Who Gets Help First](https://github.com/tcusick8/DS-4320-Project-1/blob/main/Press_Release.md)

[OneDrive Data Folder](https://1drv.ms/f/c/9e42f755abca0340/IgCP8jmuoy_UQZYjd_i5tb7pAepNvVgFgtUZ0LIixIIMgXQ?e=drhuHn)

[Pipeline](https://github.com/tcusick8/DS-4320-Project-1/tree/main/Pipeline)

License?? - state name of license here and link to file in top level of repo (use normal github conventions)


## Problem Definition:
**General Problem:** Finite EMS resources for near infinite 911 calls

**Specific refined statement:** Using the NYC EMS Incident Dispatch Data (NYC Open Data), this project develops a real-time call queue ranking model that, given a set of simultaneously active, unassigned 911 EMS calls, scores and orders each call by predicted clinical urgency, using only information available at the moment of intake (call type, reported symptoms, borough, time of day, and historical severity patterns for that call category), so that the most life-threatening incidents are consistently dispatched first and dangerous wait times for Priority 1 calls are minimized.


**Rationale for refinement:** The general problem of emergency resource allocation is vast, it encompasses staffing schedules, ambulance positioning, hospital routing, and supply chain logistics. To make the project tractable, the refined statement focuses on a single, high-stakes decision point: the moment when multiple 911 calls are queued simultaneously and a dispatcher must decide which one gets an ambulance first. This framing was chosen because it is where human judgment is most fallible and where a data-driven model can deliver the clearest marginal improvement. When calls arrive one at a time, experienced dispatchers perform well; it is the concurrent surge, ten calls in ninety seconds during a Friday night in the Bronx, where misallocation is most likely and most costly. The NYC EMS Incident Dispatch Data supports this refinement directly: it contains call type codes, final incident classifications, assigned priority levels, and precise timestamps that allow reconstruction of simultaneous call queues. The target variable, whether a call escalates to or is confirmed as Priority 1 (immediate life threat), is already encoded in the dataset's final priority field, giving a clean supervised learning label. Restricting model inputs to intake-time features only (no post-dispatch information) enforces the real-world constraint that the model must make its ranking decision before any ambulance is en route, preventing data leakage and ensuring the model is deployable in production.


**Motivation:** In emergency medicine, time is the most critical resource. For cardiac arrest, the chance of survival decreases by roughly 10% for every minute without defibrillation. For stroke, every 15-minute delay in treatment costs the average patient nearly a month of healthy life. Yet across the United States, EMS systems are under mounting pressure: call volumes have risen sharply over the past decade, ambulance staffing shortages have worsened since the COVID-19 pandemic, and urban 911 centers routinely face moments where demand exceeds available units. New York City alone processes over 1.6 million EMS calls per year, more than 4,000 per day, and a significant fraction of those arrive in simultaneous clusters during peak hours, major events, and emergencies. The status quo response to this problem is a human dispatcher applying a memorized priority code matrix, a system that was designed for a different era of call volume and has not been systematically modernized for the age of machine learning. An AI-powered queue ranking model represents a low-barrier, high-impact intervention: it does not replace dispatchers, it augments them, surfacing a ranked list that a human can accept, override, or query in real time. Beyond saving individual lives, a more rational dispatch queue reduces inequities in response time across neighborhoods, reduces ambulance repositioning costs, and provides EMS leadership with a new instrument for measuring and improving dispatch quality at scale. This project lays the analytical groundwork for that system using one of the richest open EMS datasets in the world.


[When Every Second Counts, AI Now Decides Who Gets Help First](Press_Release.md)


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












