# JL Intelligence AI Financial Analysis Tool (v1.8.1)
Scanning Company Filings and Check Compliance with Gemini 3 Pro API

Developer: Jing Zhou (Margot)
Tech Stack: React, Tailwind CSS, FastAPI (Python), Google Gemini GenAI SDK
Deployment: Render (Backend), Netlify (Frontend)
🌐 Live Production Links
User Interface: https://jl-intelligence.netlify.app/
Inference Engine: https://ai-stock-analyst-us.onrender.com
1. Project Vision & Mission
The tool was conceptualized as an institutional-grade AI research prototype. The objective was to bridge the gap between raw LLM capabilities and the specific requirements of professional investment research, focusing on deep document scanning and compliance auditing.
2. Technical Architecture & SDK Implementation
We transitioned from a static REST API model to the Google GenAI SDK (v1.55.0+).
Interactions API: Leveraging the modern "Interactions" path to provide stateful reasoning and superior model routing.
Dual-Path Fallback Logic: The backend includes a fail-safe mechanism that hot-swaps to the legacy "GenerateContent" path if the primary route encounters latency.
3. The "Efficiency vs. Depth" Trade-off (Technical Note)
Query: Does the tool read the entirety of a 200+ page report?
Current Prototype Implementation: To maintain 99.9% uptime on a restricted 512MB RAM server, the current script utilizes a "High-Alpha Sampling" strategy. It prioritizes the first 10 pages (Executive Summary, Consolidated Financials, and Management Guidance) and truncates the context to 25,000 characters.
Production-Grade Scaling Path: In a commercial deployment (e.g., for ChinaAMC), we would implement one of the following two paths to handle "Infinite Depth":
RAG (Retrieval-Augmented Generation): Pre-processing the 200+ pages into a Vector Database (Pinecone/Milvus), allowing the AI to "search and retrieve" specific sections as needed without loading the entire text into RAM.
Paid Tier Infrastructure: Upgrading to a server with 8GB+ RAM would allow for the full utilization of Gemini 1.5 Pro’s 1-million-token context window, enabling the model to ingest and cross-reference the entire document simultaneously.
4. Critical Engineering Challenges & Solutions
Memory Optimization: Implemented gc.collect() and manual del object cycles to clear the heap during PDF parsing.
Diagnostic Heartbeat: Built a React monitor to manage "cold starts," ensuring zero downtime for C-suite demos.
5. Conclusion
This tool demonstrates the ability to build resilient, compliant AI systems under tight resource constraints—a core requirement for the next generation of financial technology leadership.
