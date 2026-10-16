# Training data and deployment gate

Kaizen ships with demonstration and benchmark material only.  It does not ship
with verified real employee history and must not describe any bundled data as a
real company workforce.

To train a deployment candidate, provide a legally approved, de-identified
historical CSV.  Each row must represent information available at a documented
snapshot date and the target must mean a future outcome, for example
`AttritionWithin6Months`.  Do not include employee identifiers, free-text
manager notes, exit information, disciplinary data, medical information, or
protected characteristics as model features.

Before operational use, require all of the following:

1. A later-period holdout or a separate business-unit validation cohort.
2. A documented target window, source owner, retention period, and data-use approval.
3. Calibration, error, and disparity diagnostics reviewed with sufficiently large cohorts.
4. Privacy, security, HR, legal, and employee-representation review appropriate to the jurisdiction.
5. A human-led intervention policy that prohibits use for hiring, firing, compensation, promotion, discipline, or automated action.
6. Ongoing drift, outcome, and intervention-effect monitoring.

Use the explicit provenance requirement when training:

```powershell
python train_production_model.py --data C:\approved\attrition_history.csv --target Attrition --provenance "HRIS export, Jan 2023-Dec 2025; de-identified; approved by HR and Privacy"
```

The application can flag a probability for review. It cannot establish why an
employee will leave or prove that a proposed intervention will retain them.
