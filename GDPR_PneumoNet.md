# 🔒 GDPR & Medical AI Compliance Document
## PneumoNet AI — Chest X-Ray Classification System
### Prepared in accordance with UK GDPR, Data Protection Act 2018, and MHRA SaMD Guidance

**Document Version:** 1.0  
**Classification:** Portfolio / Educational — For review purposes only  
**Applicable Regulation:** UK GDPR, DPA 2018, MHRA Medical Device Regulations 2002  
**Supervisory Authority:** ICO (data), MHRA (medical device)

---

## 1. Executive Summary

PneumoNet AI is a deep learning image classification system for detecting pneumonia from chest X-ray images. This document demonstrates the data protection and regulatory compliance framework that would govern deployment of this type of AI system in a real UK healthcare setting.

**Current portfolio status:** No real patient data is used. The demo application generates synthetic images. UK GDPR obligations are not currently triggered. This document serves as a compliance roadmap for production deployment.

---

## 2. Nature of Data Processed

### Special Category Health Data (Article 9)

Chest X-ray images are classified as **health data** — a special category of personal data — under Article 9 of UK GDPR because:

1. **Direct health indication:** X-rays reveal detailed information about a person's physical health, anatomy, and medical conditions
2. **Potential for identification:** Some imaging formats (DICOM) contain embedded patient metadata; even without metadata, certain physical characteristics visible in imaging may enable identification
3. **Medical confidentiality:** Obtained in a clinical context with a reasonable expectation of privacy

**Consequence:** Processing chest X-rays for ML purposes requires compliance with Article 9 in addition to Article 6. Standard legitimate interests or contract bases are insufficient. The organisation must rely on one of the Article 9(2) exemptions:

| Exemption | Article | Applicability to PneumoNet |
|-----------|---------|---------------------------|
| Healthcare provision | 9(2)(h) | Applicable if system is used directly in patient care |
| Scientific or historical research | 9(2)(j) | Applicable for model development/training |
| Public interest | 9(2)(i) | Potentially applicable for NHS-funded research |

**Recommended basis for training:** Article 9(2)(j) — scientific research, combined with appropriate safeguards (pseudonymisation, access controls, ethics committee approval).

**Recommended basis for clinical inference:** Article 9(2)(h) — healthcare provision, relying on the patient's implied consent to their data being used for their own care.

---

## 3. DICOM De-Identification Requirements

All patient X-ray data used for model training must be de-identified. DICOM (Digital Imaging and Communications in Medicine) files contain extensive metadata in headers, including:

**Identifiers that must be removed or replaced:**
- Patient name, date of birth, patient ID
- Referring physician name and ID
- Accession number, study ID, series number
- Institution name, hospital department
- Dates of study, image acquisition
- Device serial numbers, operator names

**De-identification standard:** PS 3.15 DICOM Standard — Attribute Confidentiality Profiles

**Tools:**
- GDCM (Grassroots DICOM) — open-source de-identification library
- Orthanc DICOM server — includes de-identification workflow
- CTP (Clinical Trials Processor) — widely used for research de-identification

**Important caveat:** Even after header de-identification, some researchers have demonstrated re-identification from the image content itself using facial reconstruction from skull anatomy visible in head CT/MRI. For chest X-rays this risk is lower, but cannot be completely eliminated.

---

## 4. Data Protection Impact Assessment (Mandatory — Article 35)

A DPIA is **mandatory** before processing patient X-rays for AI purposes because:
- Large-scale processing of special category health data (Article 35(3)(b))
- Use of new technology (AI/ML) in a healthcare context
- Systematic evaluation of individuals (Article 35(3)(a))

### DPIA Summary Framework

**Step 1: Describe the processing**
- What data: De-identified chest X-ray images + binary diagnosis labels
- How collected: From PACS (Picture Archiving and Communication System) with ethics committee approval
- By whom: AI development team (data scientists + clinicians)
- Retention: Locked storage for duration of study + 10 years (research data)

**Step 2: Assess necessity and proportionality**
- Is AI needed? Yes — AI-assisted screening can improve diagnostic throughput and reduce workload
- Is chest X-ray the minimum necessary data? Yes — sufficient for pneumonia detection without full CT
- Is the dataset the minimum necessary? Yes — minimum sample required for statistical validity

**Step 3: Identify and assess risks**
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Re-identification from X-ray | Low | High | Strict access controls, aggregated reporting only |
| Bias leading to worse care for minorities | Medium | High | Demographic subgroup analysis, diverse training data |
| Model error causing missed diagnosis | Medium | Very high | Human oversight mandatory, audit trail |
| Data breach exposing patient images | Low | Very high | Encryption, access controls, penetration testing |
| Scope creep — use for other conditions | Low | Medium | Data use agreements, governance board oversight |

**Step 4: Measures to address risks**
- Pseudonymise all training data
- Ethics committee approval and Research Ethics Committee (REC) registration
- Data sharing agreement with NHS Trust
- Restrict access to named individuals only
- Annual data deletion/review
- Mandatory human review of all model outputs before clinical action

---

## 5. MHRA Software as a Medical Device (SaMD) Analysis

### Is PneumoNet a Medical Device?

Under the MHRA's Medical Devices Regulations 2002 and the UK Medical Device Regulations 2002 (as amended), software is a medical device if it is:
- Intended to be used for a medical purpose
- Including diagnosis, monitoring, or treatment of a disease

**Assessment:**
If PneumoNet is used to assist radiologists in diagnosing pneumonia from X-rays, it would likely be classified as a **Medical Device** — specifically Software as a Medical Device (SaMD).

### Classification

Following the IMDRF SaMD framework and MHRA guidance:

| Criterion | Assessment |
|-----------|-----------|
| State of healthcare situation | Serious (pneumonia can be life-threatening) |
| Healthcare decision significance | Diagnosis — directly influences treatment decision |
| Intended use | To inform/support, not replace, clinical judgement |
| **Likely classification** | **Class IIa — Medium risk** |

### What Class IIa Requires

1. **Technical Documentation:** Architecture diagrams, training methodology, validation data, performance evidence
2. **Quality Management System:** ISO 13485 compliance
3. **Conformity Assessment:** Assessment by a UK Approved Body (UKAB)
4. **UK Declaration of Conformity:** Legal declaration that the device meets regulations
5. **MHRA Registration:** Register the device and your organisation with MHRA
6. **Post-Market Surveillance:** Ongoing monitoring of real-world performance
7. **Adverse Event Reporting:** Report serious incidents to MHRA within 72 hours

### This Portfolio Project Avoids MHRA Regulation By:

- Explicitly labelling itself as educational/portfolio software
- Including a prominent medical disclaimer on every page
- Not being intended for use in clinical decision-making
- Not being distributed to healthcare providers
- Requiring no clinical action based on its output

---

## 6. NHS-Specific Requirements

If deployed within NHS infrastructure, additional requirements apply:

### Data Security and Protection Toolkit (DSPT)

Any system processing NHS patient data must comply with the DSPT — the NHS's framework for data security. Annual self-assessment required, covering:

- Leadership and governance
- People responsibilities
- Training
- Managing data access
- Process reviews
- Responding to incidents
- Continuity planning
- Unsupported systems
- IT protection
- Accountable suppliers

### NHS AI Lab — Evidence Standards Framework

NICE (National Institute for Health and Care Excellence) and the NHS AI Lab have published the Evidence Standards Framework for Digital Health Technologies. AI diagnostic tools are evaluated at three tiers:

| Tier | Requirement | For PneumoNet |
|------|-------------|---------------|
| 1 (Demonstrable value) | Theoretical rationale and pathway | Required |
| 2 (Clinical validity) | Published evidence of diagnostic accuracy | Required before piloting |
| 3 (Clinical utility) | Evidence of improved patient outcomes | Required before commissioning |

### Caldicott Principles

The Caldicott Principles govern use of patient information in health and social care in England. Key principles relevant to PneumoNet:

1. Justify the purpose — clear clinical or research rationale required
2. Only use personal confidential data when absolutely necessary
3. Use the minimum necessary personal confidential data
4. Access should be on a strict need-to-know basis
5. Everyone must understand their responsibilities
6. Comply with the law
7. The duty to share information can be as important as the duty to protect it
8. Inform patients and service users how their information is used

---

## 7. Bias and Fairness in Medical AI

A critical compliance consideration for medical AI is demographic bias — the risk that the model performs differently across patient groups.

**Known sources of bias in chest X-ray AI:**
- Training datasets from specific hospital systems may not generalise to other demographics
- Underrepresentation of certain ethnic groups leads to reduced performance for those groups
- Equipment and imaging protocol variation between hospitals affects image characteristics
- Age distribution in training data may not match target population

**Required fairness analysis before deployment:**
1. Compute accuracy, sensitivity, and specificity separately for: age groups, sex, ethnicity, scanner manufacturer, image acquisition site
2. Test for statistically significant performance differences between subgroups
3. If differences are found: investigate root cause and address in training data or model design
4. Document all subgroup analyses in DPIA and technical documentation

---

## 8. Incident Response

If a patient were harmed due to an incorrect model prediction:

1. **Immediate:** Withdraw the system from clinical use pending investigation
2. **Within 24 hours:** Notify the Data Protection Officer and Caldicott Guardian
3. **Within 72 hours:** Report to MHRA as a serious incident (if registered as medical device)
4. **Within 72 hours:** If personal data was involved in a breach, notify the ICO
5. **Within 72 hours:** Notify the NHS Trust's incident management team
6. **Within 30 days:** Patient notification if their data was involved
7. **Investigation:** Root cause analysis — was it a model error, a deployment error, or a user error?

---

*This document was prepared for portfolio purposes. It is not a legal document and does not constitute legal, regulatory, or medical device compliance advice. Any real deployment requires qualified legal, regulatory, and clinical expertise.*
