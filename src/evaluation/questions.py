"""
30 fixed evaluation questions with ground-truth answers.
Used to measure: answer correctness, hallucination rate, and latency.

Format:
    {
        "id": int,
        "question": str,
        "ground_truth": str,   # expected key facts / answer
        "category": str,       # topic area for grouped reporting
    }
"""

TEST_QUESTIONS = [
    # ── Customer Complaints & Trends ──────────────────────────────────────────
    {
        "id": 1,
        "question": "What were the top 3 product categories with the most customer complaints in Q3?",
        "ground_truth": "Mobile App, Billing, Shipping",
        "category": "complaints",
    },
    {
        "id": 2,
        "question": "What is the overall customer satisfaction score for Q3?",
        "ground_truth": "3.8 out of 5",
        "category": "complaints",
    },
    {
        "id": 3,
        "question": "Which complaint type increased the most compared to Q2?",
        "ground_truth": "Billing disputes increased by 22%",
        "category": "complaints",
    },
    {
        "id": 4,
        "question": "What percentage of complaints were resolved within 24 hours in Q3?",
        "ground_truth": "67%",
        "category": "complaints",
    },
    {
        "id": 5,
        "question": "Which region had the highest volume of complaints in Q3?",
        "ground_truth": "North America",
        "category": "complaints",
    },
    # ── Product Improvement ───────────────────────────────────────────────────
    {
        "id": 6,
        "question": "What are the top 3 product improvements requested by customers?",
        "ground_truth": "Faster checkout, better mobile UX, transparent billing",
        "category": "product",
    },
    {
        "id": 7,
        "question": "How many unique feature requests were logged in Q3?",
        "ground_truth": "142 unique feature requests",
        "category": "product",
    },
    {
        "id": 8,
        "question": "Which product feature had the lowest NPS score?",
        "ground_truth": "Mobile app checkout flow, NPS -12",
        "category": "product",
    },
    {
        "id": 9,
        "question": "What is the most common root cause of shipping complaints?",
        "ground_truth": "Third-party logistics partner delays",
        "category": "product",
    },
    {
        "id": 10,
        "question": "What improvement had the highest potential ROI according to the report?",
        "ground_truth": "Automated billing reconciliation, projected 18% cost reduction",
        "category": "product",
    },
    # ── Policy & Compliance ───────────────────────────────────────────────────
    {
        "id": 11,
        "question": "What is the company's refund policy for digital purchases?",
        "ground_truth": "30-day refund window for unused digital products",
        "category": "policy",
    },
    {
        "id": 12,
        "question": "What is the escalation procedure for unresolved complaints?",
        "ground_truth": "Tier 1 → Tier 2 within 48h → Manager review within 5 business days",
        "category": "policy",
    },
    {
        "id": 13,
        "question": "Are customers notified when their complaint is escalated?",
        "ground_truth": "Yes, via email and in-app notification",
        "category": "policy",
    },
    {
        "id": 14,
        "question": "What is the SLA for critical billing complaints?",
        "ground_truth": "Resolution within 4 business hours",
        "category": "policy",
    },
    {
        "id": 15,
        "question": "Which data privacy regulation governs customer complaint data storage?",
        "ground_truth": "GDPR (EU customers) and CCPA (California customers)",
        "category": "policy",
    },
    # ── Strategy & Insights ───────────────────────────────────────────────────
    {
        "id": 16,
        "question": "What is the projected impact of improving checkout speed on conversion rate?",
        "ground_truth": "Estimated 8-12% increase in conversion rate",
        "category": "strategy",
    },
    {
        "id": 17,
        "question": "How does Q3 complaint volume compare to the same period last year?",
        "ground_truth": "Up 14% year-over-year",
        "category": "strategy",
    },
    {
        "id": 18,
        "question": "What retention risk is associated with unresolved billing complaints?",
        "ground_truth": "Customers with unresolved billing issues are 3x more likely to churn",
        "category": "strategy",
    },
    {
        "id": 19,
        "question": "Which customer segment files the most complaints per order?",
        "ground_truth": "Enterprise accounts, 2.1 complaints per 100 orders",
        "category": "strategy",
    },
    {
        "id": 20,
        "question": "What is the average resolution cost per complaint?",
        "ground_truth": "$42 per complaint including agent time and credits",
        "category": "strategy",
    },
    # ── Operational Metrics ───────────────────────────────────────────────────
    {
        "id": 21,
        "question": "What is the average handle time for Tier 1 support tickets?",
        "ground_truth": "8.3 minutes",
        "category": "operations",
    },
    {
        "id": 22,
        "question": "How many support agents were active in Q3?",
        "ground_truth": "247 agents across 3 time zones",
        "category": "operations",
    },
    {
        "id": 23,
        "question": "What channel has the highest complaint volume?",
        "ground_truth": "Email (43%), followed by in-app chat (31%)",
        "category": "operations",
    },
    {
        "id": 24,
        "question": "What percentage of complaints were categorised as 'billing errors'?",
        "ground_truth": "28% of all complaints",
        "category": "operations",
    },
    {
        "id": 25,
        "question": "What is the first-contact resolution rate for Q3?",
        "ground_truth": "54%",
        "category": "operations",
    },
    # ── Competitor & Market ───────────────────────────────────────────────────
    {
        "id": 26,
        "question": "How does our customer satisfaction score compare to industry benchmark?",
        "ground_truth": "3.8 vs industry average of 4.1, 0.3 below benchmark",
        "category": "market",
    },
    {
        "id": 27,
        "question": "What benchmark metric is most critical to close the satisfaction gap?",
        "ground_truth": "First-contact resolution rate; industry benchmark is 72% vs our 54%",
        "category": "market",
    },
    # ── Multi-step Synthesis ──────────────────────────────────────────────────
    {
        "id": 28,
        "question": "Summarise Q3 customer complaint trends and suggest the top 3 product improvements.",
        "ground_truth": (
            "Top trends: billing disputes (+22%), mobile app UX issues, shipping delays. "
            "Top improvements: automated billing reconciliation, mobile checkout redesign, "
            "logistics partner SLA enforcement."
        ),
        "category": "synthesis",
    },
    {
        "id": 29,
        "question": (
            "Which product areas should the engineering team prioritise in Q4 "
            "to reduce complaint volume by at least 15%?"
        ),
        "ground_truth": (
            "Billing automation (addresses 28% of complaints), "
            "mobile app performance (addresses 24%), "
            "shipping tracking transparency (addresses 19%)."
        ),
        "category": "synthesis",
    },
    {
        "id": 30,
        "question": (
            "Draft an executive summary of Q3 support performance, "
            "highlighting risks and recommended actions."
        ),
        "ground_truth": (
            "Executive summary should mention: 14% YoY complaint growth, "
            "3x churn risk from unresolved billing, "
            "FCR rate 18pp below benchmark, "
            "recommend billing automation and mobile UX investment."
        ),
        "category": "synthesis",
    },
]
