# MLOps OPPE-2 AI Usage Documentation

**Student Name:** Vishwas Mehta  
**Roll Number:** 22F3001150  
**Assignment:** Heart Disease Prediction - MLOps Pipeline  
**Date:** December 14, 2025

---

## AI Tools Utilized and Conversation History

> I used AI tools primarily for debugging, syntax clarification, and configuration assistance. The core implementation, model training, analysis, and architectural decisions were done independently.

### Tool 1: Perplexity AI
- **Purpose:** Debugging Kubernetes configuration issues, Docker networking troubleshooting, and GitHub Actions workflow syntax validation
- **Shared Chat Link:** [Available upon request - extensive conversation history]
- **Notes:** Used for specific error resolution and best practices verification, not for generating complete solutions

### Tool 2: Stack Overflow & Documentation
- **Purpose:** Understanding SHAP library documentation, Fairlearn API usage examples, and Evidently drift detection
- **Shared Chat Link:** N/A (Public documentation)
- **Notes:** Used official documentation and community examples for library-specific syntax clarification

---

## Prompts and Responses Used

> Below are the key areas where AI assistance was utilized. All assistance was for clarification, debugging, or syntax verification.

---

### Perplexity AI - Kubernetes Debugging

**Prompt 1:**


**Response Summary:**  
Received HPA YAML configuration example with `metrics` section showing both CPU and memory targets. Used this as reference to create my own `hpa.yaml`.

**My Implementation:**  
Modified the example to create `hpa.yaml` with `maxReplicas: 3`, CPU threshold at 50%, and memory threshold at 70% based on assignment requirements.

---

**Prompt 2:**



**Response Summary:**  
Learned to check for existing port-forward processes using `lsof -i :8080` and kill them with `pkill -f "port-forward"`.

**My Implementation:**  
Incorporated this check into my testing workflow, ensuring clean port-forward connections before running performance tests.

---

### Documentation Research - SHAP Library

**Prompt 3:**



**Response Summary:**  
Learned to use boolean indexing on SHAP values array with misclassified indices, then apply `.abs().mean(axis=0)` to get average feature impact.

**My Implementation:**  
Applied this in Cell 5 of my notebook to analyze the 9 misclassified samples and identify top contributing features: CP (1.02), oldpeak (0.76), gender (0.62), ca (0.52), thal (0.36).

---

**Prompt 4:**



**Response Summary:**  
Understood that age bins need to be created using `pd.cut()` before passing to MetricFrame as the sensitive_features parameter.

**My Implementation:**  
Created three age bins: Young (≤40), Middle (41-60), Senior (>60), then computed fairness metrics including demographic parity difference (0.24) and equalized odds difference (0.39).

---

### Perplexity AI - GitHub Actions Configuration

**Prompt 5:**



**Response Summary:**  
Got workflow structure with `google-github-actions/auth@v1` and `google-github-actions/setup-gcloud@v1` actions. Learned about storing service account JSON in GitHub Secrets.

**My Implementation:**  
Created complete CI/CD pipeline with 3 jobs (test, build-and-push, deploy) adapting the authentication pattern to my GCP project (adroit-metric-473508-k1) and cluster configuration.

---

**Prompt 6:**



**Response Summary:**  
Learned wrk uses Lua scripts for complex requests with `wrk.method`, `wrk.body`, and `wrk.headers` syntax.

**My Implementation:**  
Created `wrk_payload.lua` with heart disease prediction JSON payload and executed performance test with `-t 4 -c 2000 -d 30s` parameters to meet assignment requirements.

---

### Documentation Research - Evidently Drift Detection

**Prompt 7:**

This is the promt

**Response Summary:**  
Understood the structure: create `Report` with `DataDriftPreset()`, run with reference and current data, then save as HTML.

**My Implementation:**  
Applied this to compare training data (234 samples) vs random test data (100 samples), generated `drift_report.html`, and analyzed features showing significant drift (thalach, oldpeak, gender).

---

**Prompt 8:**


**Response Summary:**  
Learned about `/health` endpoint patterns and proper JSON error responses with appropriate HTTP status codes.

**My Implementation:**  
Created Flask API with `/health` returning model status and `/predict` with try-catch error handling, proper logging, and structured JSON responses.

---

## What I Did Independently (Without AI Assistance)

### 1. Problem Analysis & Approach Design
- Read and understood all 8 deliverables of the assignment
- Designed the overall MLOps pipeline architecture
- Decided on technology stack: Flask, Docker, Kubernetes, GitHub Actions, SHAP, Fairlearn, Evidently

### 2. Data Preprocessing & Model Training
- Analyzed heart disease dataset structure and characteristics
- Decided preprocessing strategy: dropping serial numbers, binary encoding for gender and target
- Handled missing values by dropping NaN rows (234 clean samples)
- Chose 80-20 train-test split with stratification
- Trained RandomForestClassifier with appropriate parameters based on course knowledge

### 3. SHAP Explainability Analysis (Deliverable 3)
- Implemented SHAP analysis workflow independently
- **Interpreted results independently:** Identified CP (chest pain) as the most important feature with 1.02 impact
- Analyzed why misclassifications occur based on feature patterns
- Wrote plain English explanation: "Chest pain characteristics and cardiac stress markers (oldpeak) are primary factors affecting misclassifications"

### 4. Fairness Testing Analysis (Deliverable 4)
- Decided on age binning strategy: ≤40, 41-60, >60 years
- **Interpreted fairness metrics independently:** Demographic parity 0.24, Equalized odds 0.39
- Made judgment call: "Model shows acceptable fairness without severe discrimination"
- Considered real-world implications for medical screening

### 5. Flask API Development
- Designed REST API structure with two endpoints
- Implemented prediction logic with probability calculation
- Added comprehensive error handling and logging
- Wrote structured responses with input echoing for debugging

### 6. Docker Configuration
- Created Dockerfile with Python 3.9 slim base image
- Configured proper working directory, dependency installation, and port exposure
- Added health check configuration for Kubernetes liveness probes
- Tested locally before cloud deployment

### 7. Kubernetes Manifests
- Wrote `deployment.yaml` with resource limits, requests, and health probes
- Created `service.yaml` for load balancing
- Designed `hpa.yaml` with scaling thresholds based on assignment requirements (max 3 pods)
- Understood the relationship between deployments, services, and HPAs

### 8. CI/CD Pipeline Design
- Structured 3-stage pipeline: test → build-and-push → deploy
- Configured GCP authentication with service account
- Set up conditional job execution (deploy only on main branch push)
- Implemented zero-downtime deployment with `kubectl rollout restart`

### 9. Logging & Observability Implementation
- Created Python script for structured prediction logging
- Implemented CSV log storage for analysis
- Integrated with GCP Cloud Logging for production monitoring
- Logged 10 sample predictions with age, prediction, and confidence

### 10. Performance Testing Execution
- Ran wrk performance test with 2000+ concurrent connections
- **Analyzed results independently:** 1.34s avg latency, 51.98 req/sec, 1510 timeouts
- **Explained why timeouts occurred:** Single pod handling extreme load
- **Made recommendation:** HPA will scale to 3 pods under real load, improving performance

### 11. Drift Detection Analysis
- Executed Evidently drift detection on 100 random samples vs training data
- **Interpreted drift report independently:** Identified features with detected drift (thalach, oldpeak, gender)
- **Made judgment:** "Moderate drift, not severe - model remains usable but should be monitored"
- Understood implications for production model maintenance

### 12. GCP Infrastructure Setup
- Created GKE cluster named `iris-cluster` in us-central1
- Set up Artifact Registry repository `mlops-models` in asia-south1
- Configured service account with appropriate IAM roles
- Generated and secured service account key for GitHub Actions

### 13. Repository Organization
- Structured folders: `deployment/`, `k8s/`, `.github/workflows/`, `artifacts/`, `data/`
- Created comprehensive README with setup instructions
- Added `.gitignore` to exclude credentials and temporary files
- Made repository private and added IITMBSMLOps as collaborator

### 14. Testing & Validation
- Tested Docker container locally before cloud deployment
- Validated Kubernetes deployment with health checks
- Verified CI/CD pipeline execution through GitHub Actions
- Confirmed all 8 deliverables are complete and functional

---

## Summary

**AI Usage Pattern:**  
AI tools were used **strategically for specific technical queries**, particularly for:
- Kubernetes YAML syntax and configuration patterns
- Library-specific API usage (SHAP, Fairlearn, Evidently)
- GitHub Actions GCP authentication workflow
- Command-line tools syntax (wrk, kubectl, docker)
- Debugging specific errors encountered during implementation

**Independent Work:**  
All substantive work was completed independently, including:
- Problem analysis and solution design
- Model training, evaluation, and hyperparameter selection
- **Critical analysis and interpretation** of SHAP, fairness, and drift results
- Architecture design for the MLOps pipeline
- End-to-end implementation and integration
- Testing, debugging, and deployment
- Documentation and video demonstration

**Percentage Estimate:**  
Approximately **15-20% assistance** for syntax/debugging, **80-85% independent work** for design, implementation, analysis, and interpretation.

---

## Declaration

I declare that while I used AI tools and online resources for technical clarification and debugging as documented above, the substantial work including problem understanding, approach design, model training, result analysis, interpretation, architectural decisions, and implementation was completed by me independently. 

The AI usage was limited to resolving specific technical queries and syntax validation. All analytical thinking, decision-making, and interpretation of results represent my own understanding and work.

I understand the importance of academic integrity and have documented all AI assistance transparently.

**Student Name:** Vishwas Mehta  
**Roll Number:** 22F3001150  
**Date:** December 14, 2025

---
Here is the thing
## Appendix: Tools and Resources Used

### AI Tools
1. Perplexity AI - For debugging and configuration assistance
2. ChatGPT (limited) - For documentation clarification

### Official Documentation
1. Kubernetes Documentation - https://kubernetes.io/docs/
2. Docker Documentation - https://docs.docker.com/
3. GitHub Actions Documentation - https://docs.github.com/actions
4. SHAP Documentation - https://shap.readthedocs.io/
5. Fairlearn Documentation - https://fairlearn.org/
6. Evidently AI Documentation - https://docs.evidentlyai.com/
7. GCP Documentation - https://cloud.google.com/docs

### Community Resources
1. Stack Overflow - For specific error resolution
2. GitHub Issues - For library-specific troubleshooting

---

**End of AI Usage Documentation**



