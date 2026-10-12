# Hackathon Brief

**PredictLeads × TripleTen · Job Seeker Intelligence Platform · 3-Week Hackathon | DS & AI/ML Teams**

> The official project brief, transcribed from the organizers' PDF (kept locally, not in git). Only the S3 location is left out: it's in `S3_PATH` in your `.env` (README step 8). Everything else is as written. Where our plans differ from it, the brief wins unless the team decides otherwise and logs it in [`DECISION_LOG.md`](analysis/DECISION_LOG.md).

## About the partner

PredictLeads provides company intelligence data covering job openings, financing, technology adoption, news and other company signals. For this project, students will primarily work with PredictLeads' job-opening data, with the potential to enrich their solution with additional company-level signals.

## What you are building

Build an **ML- and GenAI-powered job seeker platform** that helps users discover and prioritize relevant job opportunities based on their skills, experience, target roles, location and preferences.

The platform should include:
- Job search and filtering
- ML-based job-candidate matching/ranking
- Personalized job recommendations
- GenAI-powered explanations (e.g. a clear explanation of why a job is a good match) and insights
- At least one additional GenAI capability, such as:
  - Job summaries
  - Skills-gap analysis
  - Personalized application recommendations
  - Candidate/job profile summarization

## Candidate profiles

Teams do not need to use their own resumes. They may use their own resumes if they are comfortable, or alternatively:
- Source publicly available sample resumes
- Create synthetic candidate profiles
- Generate realistic candidate profiles using GenAI

Teams should document how candidate profiles were sourced or created and consider privacy, consent and data quality. If real resumes are used, teams must appropriately identify and mask/remove PII before using the data in their application or sharing it with external services.

**Optional:** Teams may explore automated PII detection and redaction as an additional technical component.

## Data

PredictLeads will provide two datasets for the hackathon:
- **Job Openings:** a six-month dataset of US job openings, including job lifecycle information such as `first_seen_at` and `last_seen_at`, allowing teams to analyze job activity over time and reconstruct historical job-market states. File: `job_openings_2026_09_30.jsonl.zst` (~17.7 GB compressed). Documentation: https://docs.predictleads.com/guide/job_openings_dataset
- **Companies:** a company-level dataset that can be used to enrich job opportunities with additional company information, such as company descriptions and domain names. File: `companies_2026_09_30.jsonl.zst` (~240 MB compressed). Documentation: https://docs.predictleads.com/guide/companies_dataset

## Data access

The datasets are stored in a private AWS S3 bucket (folder `2026/09/30/`; the bucket is in `S3_PATH` in `.env`). Teams will receive read-only AWS credentials **separately**. Teams may access the data programmatically using AWS CLI, Python (`boto3`), or other appropriate tools.

Because the Job Openings dataset is large, teams should use appropriate streaming, chunking, filtering, or sampling approaches rather than loading the entire dataset into memory at once.

**Data security:** AWS credentials must be kept confidential and must not be committed to GitHub, included in project code, or shared outside the project. PredictLeads data must only be used for this project and must not be publicly distributed.

Teams should use the provided PredictLeads datasets as the core data source for their ML and application development. Teams may combine the PredictLeads datasets with candidate profiles created or sourced according to the guidelines above.

## 3-week roadmap

### Week 1: Data & ML Foundation

**Goal:** Understand the data, define the matching problem and build the first working ML approach.

Teams should:
- Explore and clean the job-opening dataset
- Explore `first_seen_at` / `last_seen_at` and determine how job activity will be represented over time
- Define the candidate profile and matching problem
- Decide how candidate profiles will be sourced/created
- Define evaluation metrics
- Build a baseline ML matching/ranking model
- Define the GenAI use cases they will implement

**End-of-week deliverable:** Structured dataset, candidate profiles, documented modeling, evaluation and GenAI approach, baseline matching/ranking model and initial evaluation results.

### Week 2: Product & AI Build

**Goal:** Turn the ML model into a usable product.

Teams should:
- Build the core user flow and application
- Integrate the ML matching/ranking model
- Implement job search and recommendations
- Integrate required GenAI functionality
- Add explainability around recommendations
- Integrate GenAI features: connect the ML and GenAI components into the user experience

**End-of-week deliverable:** A functional MVP with the core user journey working end-to-end, including both ML matching/ranking and GenAI functionality.

### Week 3: Refinement & Final Demo

**Goal:** Turn the MVP into a polished, portfolio-ready product.

Teams should:
- Improve model performance and ranking
- Test the application with multiple candidate profiles
- Evaluate the quality of recommendations and GenAI outputs
- Refine UX and AI outputs
- Document the technical approach, evaluation and limitations
- Prepare the final presentation and demo
- Suggest a roadmap for the product

**End-of-project deliverables:**
- Working application/demo
- ML model and evaluation results
- GenAI functionality
- Source code/repository
- Short technical/project README
- Product roadmap outlining future improvements, features and potential next steps
- Final presentation

## Tools

Students may use:
- **Python and associated libraries:** data processing & ML
- **LLM APIs / Hugging Face:** GenAI (open-source)
- **Streamlit / React / Vercel / similar:** application UI
- **GitHub:** version control
- **AWS S3:** data access

Note: Teams may use low-code/no-code or AI-assisted tools to build the application UI. The ML pipeline, data processing, matching/ranking logic and GenAI integrations must be implemented and documented by the team.

## Out of scope

- Building a production-scale job board
- Real-time job ingestion or continuous data pipelines
- Predicting whether a candidate will be hired
- Resume generation as the primary objective
