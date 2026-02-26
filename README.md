## AquaBa-AI Data Science Team

### Project Overview
This repository contains the **AI-Enhanced Supply Chain Optimizer** components for seafood SMEs.  
Two main notebooks are currently available:

- **Sp9.ipynb** → Spoilage Alert System (rule-based + anomaly detection).  
- **Sp10_Preprocessing.ipynb** → Synthetic demand dataset generation + preprocessing for forecasting.

---

### Spoilage Alerts (Sp9.ipynb)
**Owner:** Jane + Barakat  
**Purpose:** Detect spoilage risks using sensor data.  

**How to Run:**
1. Open `Sp9.ipynb` in Colab or Jupyter.  
2. Run all cells to generate synthetic sensor data.  
3. The notebook outputs:
   - Rule-based spoilage alerts (NORMAL / WARNING / CRITICAL).  
   - Anomaly detection alerts (z-score / isolation forest).  
   - Combined alert messages.  
4. Exported results can be saved as CSV for testing integration.  

---

###  Forecasting Preprocessing (Sp10_Preprocessing.ipynb)  
**Purpose:** Generate synthetic demand data and prepare it for LSTM forecasting.  

**How to Run:**
1. Open `AquaBa Ai practice1.ipynb` in Colab or Jupyter.  
2. Run all cells to:
   - Generate synthetic demand dataset with **seasonality + port-specific differences**.  
   - Apply preprocessing (cleaning, scaling, lag features, rolling averages).  
   - Split into train/test sets.  
3. The notebook exports two files:
   - `train_demand.csv` → training dataset for LSTM.  
   - `test_demand.csv` → testing dataset for LSTM.  



Would you like me to also **draft the README section for Sp11_Forecasting.ipynb** (your upcoming LSTM notebook), so it’s already prepared when you commit the file?
