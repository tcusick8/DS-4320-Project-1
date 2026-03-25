# Project Details
DS-4320-Project-1: ----Project Title-----
¿¿(Emergency Medical Services Analysis and Dataset building for Project 1 in DS 4320 Data by Design??)
----Executive sumary, short paragraph explainaing contents of repo in executive form-----

Thomas Cusick
tpg6hu
DOI ------
(Press Release)[---------]
(OneDrive Data Folder)[https://1drv.ms/f/c/9e42f755abca0340/IgCP8jmuoy_UQZYjd_i5tb7pAepNvVgFgtUZ0LIixIIMgXQ?e=drhuHn]
(Pipeline)[-------]
License?? - state name of license here and link to file in top level of repo (use normal github conventions)

## Problem Definition:
General Problem: Finite EMS resources for near infinite 911 calls

## Specific refined statement: Using the NYC EMS Incident Dispatch Data (NYC Open Data), this project develops a real-time call queue ranking model that, given a set of simultaneously active, unassigned 911 EMS calls, scores and orders each call by predicted clinical urgency, using only information available at the moment of intake (call type, reported symptoms, borough, time of day, and historical severity patterns for that call category), so that the most life-threatening incidents are consistently dispatched first and dangerous wait times for Priority 1 calls are minimized.

## Rationale for refinement: The general problem of emergency resource allocation is vast, it encompasses staffing schedules, ambulance positioning, hospital routing, and supply chain logistics. To make the project tractable, the refined statement focuses on a single, high-stakes decision point: the moment when multiple 911 calls are queued simultaneously and a dispatcher must decide which one gets an ambulance first. This framing was chosen because it is where human judgment is most fallible and where a data-driven model can deliver the clearest marginal improvement. When calls arrive one at a time, experienced dispatchers perform well; it is the concurrent surge, ten calls in ninety seconds during a Friday night in the Bronx, where misallocation is most likely and most costly. The NYC EMS Incident Dispatch Data supports this refinement directly: it contains call type codes, final incident classifications, assigned priority levels, and precise timestamps that allow reconstruction of simultaneous call queues. The target variable, whether a call escalates to or is confirmed as Priority 1 (immediate life threat), is already encoded in the dataset's final priority field, giving a clean supervised learning label. Restricting model inputs to intake-time features only (no post-dispatch information) enforces the real-world constraint that the model must make its ranking decision before any ambulance is en route, preventing data leakage and ensuring the model is deployable in production.

## Motivation: In emergency medicine, time is the most critical resource. For cardiac arrest, the chance of survival decreases by roughly 10% for every minute without defibrillation. For stroke, every 15-minute delay in treatment costs the average patient nearly a month of healthy life. Yet across the United States, EMS systems are under mounting pressure: call volumes have risen sharply over the past decade, ambulance staffing shortages have worsened since the COVID-19 pandemic, and urban 911 centers routinely face moments where demand exceeds available units. New York City alone processes over 1.6 million EMS calls per year, more than 4,000 per day, and a significant fraction of those arrive in simultaneous clusters during peak hours, major events, and emergencies. The status quo response to this problem is a human dispatcher applying a memorized priority code matrix, a system that was designed for a different era of call volume and has not been systematically modernized for the age of machine learning. An AI-powered queue ranking model represents a low-barrier, high-impact intervention: it does not replace dispatchers, it augments them, surfacing a ranked list that a human can accept, override, or query in real time. Beyond saving individual lives, a more rational dispatch queue reduces inequities in response time across neighborhoods, reduces ambulance repositioning costs, and provides EMS leadership with a new instrument for measuring and improving dispatch quality at scale. This project lays the analytical groundwork for that system using one of the richest open EMS datasets in the world.









