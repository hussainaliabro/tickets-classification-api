# Ticket Classification API

Production-ready API that automatically analyzes and categorizes customer support tickets. It converts unstructured email content into structured metadata — Queue, Language, and Priority — enabling faster routing, consistent handling, and improved SLA compliance.

Live demo (hosted)
**https://tickets-classification-api.up.railway.app/**

Use this endpoint immediately for testing, evaluation, and rapid integration.

---

Table of contents
- [Executive summary](#executive-summary)
- [Business benefits](#business-benefits)
- [Core capabilities](#core-capabilities)
- [API specification](#api-specification)
- [Integration examples](#integration-examples)
- [System architecture and processing](#system-architecture-and-processing)
- [Deployment & reliability](#deployment--reliability)
- [Operational considerations](#operational-considerations)
- [Customization & enterprise options](#customization--enterprise-options)
- [Project layout](#project-layout)
- [Contact & next steps](#contact--next-steps)
- [Author](#author)

---

Executive summary
This Ticket Classification API is a simple, secure, and scalable service that automates triage of incoming customer-support emails. It returns three production-usable labels — which queue should handle the ticket, the detected language (English or German), and the ticket priority — enabling downstream systems to route, escalate, and assign workload automatically.

---

Business benefits
- Faster response times and improved CSAT: Automate first routing and escalate high-priority tickets immediately.  
- Reduced manual effort: Cut triaging overhead so agents focus on resolution rather than routing.  
- Consistent decisioning: Remove human inconsistency with deterministic, repeatable classification.  
- Multilingual support: Auto-detect language to enable localized workflows and SLA assignment.  
- Low integration cost: Single POST endpoint returns structured JSON you can plug into any support stack.

---

Core capabilities
- Predicts: Queue (team/queue), Language (e.g., `en`, `de`) and Priority (`low`, `medium`, `high`)  
- Inputs: `subject` and `body` text fields (JSON)  
- Fast, deterministic inference using lightweight scikit-learn models  
- Ready for immediate integration — hosted demo available at the URL above

---

API specification

Endpoint
- POST /predict
- Base URL (hosted demo): `https://tickets-classification-api.up.railway.app/`

Request schema (JSON)
```json
{
  "subject": "Laptop is not working",
  "body": "My laptop suddenly stopped working after the latest update. Please fix this issue as soon as possible."
}
```

Response schema (JSON)
```json
{
  "Queue": ["Technical Support"],
  "Language": ["en"],
  "Priority": ["high"]
}
```

Notes
- The response returns arrays for each label so the service can be extended to support multiple predictions or ensembles in the future.
- Validation: Both `subject` and `body` are required strings.

---

Integration examples

cURL (single ticket)
```bash
curl -X POST "https://tickets-classification-api.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Payment is stuck",
    "body": "I made a payment yesterday but the transaction still shows pending. Please help."
  }'
```

Python (requests)
```python
import requests

url = "https://tickets-classification-api.up.railway.app/predict"
payload = {
    "subject": "Payment is stuck",
    "body": "I made a payment yesterday but the transaction still shows pending."
}

resp = requests.post(url, json=payload)
print(resp.json())
```

Suggested integration flow
1. Send new ticket content to `/predict`.  
2. Use `Queue` to set assignment or routing rule in your ticketing system.  
3. Use `Priority` to adjust SLA, add escalation, or notify managers.  
4. Use `Language` to route to local teams or trigger translation/localized workflows.

---

System architecture and processing

High-level design
- FastAPI REST service exposes the prediction endpoint.
- A lightweight scikit-learn model is loaded at startup and serves predictions synchronously.
- Preprocessing:
  - Subject + body concatenated and normalized.
  - Semantic TF-IDF (word n-grams) with stopword removal for Queue and Priority.
  - Character n-gram TF-IDF for language detection — robust on short texts.
- Predictors: LinearSVC classifiers for Queue, Language, and Priority.

Extensibility
- Each component (preprocessing, vectorizers, classifiers) is modular and can be retrained independently without changing the API contract.
- The model artifact can be replaced with a retrained or custom model while preserving endpoint compatibility.

---

Deployment & reliability

Hosted demo
- The API is deployed on Railway at:
  https://tickets-classification-api.up.railway.app/

Advantages of the hosted deployment
- Instant access: No setup required to begin evaluation or demoing the capability.  
- Secure transport: HTTPS by default for encrypted API traffic.  
- Minimal operational overhead: No need to provision servers for quick pilots.  
- Production-ready stack: Built on FastAPI + Uvicorn for high throughput and low latency.  
- Easy migration: Same API contract can be deployed into your cloud or on-prem environment later.

Scaling & production best practices
- For production, run behind a load balancer with autoscaling and health checks.
- Containerize the app (Docker) and use replicas for high availability.
- Add authentication (API keys, OAuth) and request throttling for security and cost control.

---

Operational considerations

Security and privacy
- Demo endpoint: Avoid sending sensitive PII or regulated data to the public demo.  
- Production: Host in your environment if you require data residency or compliance guarantees. Implement transport-level security, authentication, and audit logs.

Data & model lifecycle
- Retraining: Periodically retrain on labeled historical tickets to maintain or improve accuracy.  
- Monitoring: Log predictions and track drift, accuracy, and response latency.  
- Versioning: Maintain model versions and the ability to rollback.

Dependencies
- The model and API rely on standard Python ML libraries (scikit-learn, pandas, joblib, nltk). Ensure dependencies and NLTK corpora are available in production images.

---

Customization & enterprise options
- Retrain on your historical tickets and custom taxonomies (queue and priority labels).  
- Add confidence scores and thresholding for automated escalation.  
- Integrate with enterprise ticketing systems (Zendesk, Jira, ServiceNow, Salesforce).  
- Private deployment (VPC / on-prem) with authentication, logging, and monitoring for compliance.

Typical engagement options
- Quick pilot: Plug the demo endpoint into one support queue for 2–4 weeks.  
- Custom retraining: Deliver a model tuned for your historical data and taxonomies.  
- Managed deployment: Containerized, authenticated, and monitored production deployment in your environment.

---

Project layout (high level)
ticket-classification-api/
├── app/                     # FastAPI application and model loader
│   ├── main.py              # API entrypoint
│   ├── model_loader.py      # Loads model artifact
│   └── schemas.py           # Request/response schemas
├── data/                    # Training / sample data
├── customer support tickets classification.ipynb  # Notebook used for training
├── support-tickets-classifier-model.pkl  # Pretrained model artifact
├── requirements.txt
├── Procfile
├── README.md
└── .gitignore

---

Contact & next steps
If you would like:
- A live demo integrated with your ticketing system,  
- The model retrained on your historical tickets, or  
- A private, production-grade deployment (Docker/Kubernetes) with authentication and monitoring,  

please contact:
- GitHub: @hussainaliabro

Suggested next step
- Start a 2-week pilot by routing a small proportion of incoming tickets to the API and measure improvements in time-to-first-response and triage accuracy.

---

Author
Hussain Ali Zahid  
Machine Learning Engineer
