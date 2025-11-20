# forecast-studio-interactive-time-series-dashboard

This repository contains a **full-stack forecasting dashboard** built using:

- **Frontend:** React (Material Dashboard React UI)
- **Backend:** Python (FastAPI/Flask depending on your `app.py`)
- **Models:** ARIMA, SARIMA, Naïve Seasonal Baseline, Trend Overlay
- **Visuals:** ACF, PACF, Weekly & Monthly Seasonality, Forecast Comparison

---

## Features

### Forecasting Models
- ARIMA & SARIMA forecasts  
- Configurable model parameters  
- Trend overlay  
- Naïve Seasonal Baseline (3D card)  
- Default output to avoid empty graphs  

### Diagnostics
- ACF & PACF plots  
- Weekly & Monthly seasonality  
- Forecast vs actual graph  

---

## Project Structure

Forecast Studio – Interactive Time Series Forecasting Dashboard (React + Python)/
├── amazon-forecast-backend/  
│   └── app.py  
├── material-dashboard-react/  
│   ├── public/  
│   ├── src/  
│   │   ├── App.js  
│   │   ├── components/  
│   │   ├── layouts/  
│   │   │   ├── forecast/  
│   │   │   │   └── index.js  
│   │   └── routes.js  
│   └── package.json  
└── LICENSE.md  

---

## How To Use the Dashboard

1. **Start Backend**
```
cd amazon-forecast-backend
python app.py
```

2. **Start Frontend**
```
cd material-dashboard-react
npm install
npm start
```

3. Open:  
http://localhost:3000

4. Navigate via Sidebar → **Forecast**  

5. Upload CSV → Configure model → View results  

---

## Dataset Format

date,value  
2020-01-01,120  
2020-01-02,150  

---

## License Notice

The UI template (Material Dashboard React) is MIT licensed.  
✔ You may keep and push `LICENSE.md` exactly as provided.  
✔ Fully allowed for personal, academic, and commercial usage.

---

## Credits

- UI: Material Dashboard React (Creative Tim)  
- Models: Python statsmodels  
- Dashboard built by: *Your Name*  
