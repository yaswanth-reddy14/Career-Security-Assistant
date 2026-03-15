# Career Security Assistant

Career Security Assistant is a Flask-based web application for company-level and employee-level layoff risk prediction, with OTP-verified user workflows and automated email alerts.

## Features

- Company layoff risk prediction with risk score, risk level, and cause breakdown
- Employee layoff risk prediction with role guidance and skill-gap recommendations
- Dynamic risk thresholds loaded from `risk_thresholds.json`
- OTP-based signup verification and profile email-change verification
- Automated alert delivery (SMTP) with cooldown-aware logging
- Optional MongoDB storage for users and alert logs (CSV fallback supported)
- Streamlit dashboard support (`dashboard.py`) in addition to Flask app (`app.py`)

## Project Structure

- `app.py`: Main Flask application
- `auth.py`: User/authentication storage and profile utilities
- `alerts.py`: SMTP email sending, cooldown logic, and alert logging
- `fyp.py`: Model training pipeline and threshold generation
- `dashboard.py`: Streamlit version of the interface
- `company_model.pkl`: Included trained company model
- `employee_model.pkl`: Generated locally (excluded from git because GitHub file-size limit)
- `company-level.csv`, `Employee-levelData.csv`: Source datasets
- `risk_thresholds.json`: Risk bucket thresholds
- `templates/`, `static/`: UI templates and static assets

## Requirements

- Python 3.10+
- pip

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
# App session
FLASK_SECRET_KEY=change-this-secret-key

# Optional admin bootstrap
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

# SMTP (required for real OTP/email alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_SENDER=your_email@gmail.com

# Optional MongoDB (if empty, CSV storage is used)
MONGODB_URI=
MONGODB_DB=career_security
MONGODB_USERS_COLLECTION=users
MONGODB_ALERTS_COLLECTION=alerts
MONGODB_TIMEOUT_MS=15000
```

Notes:

- For Gmail, use a 16-character App Password (not your normal Gmail password).
- If `MONGODB_URI` is blank, the app automatically uses CSV files.

## Prepare Models

Before first run, generate missing model artifacts:

```powershell
python fyp.py
```

This creates/refreshes:

- `company_model.pkl`
- `employee_model.pkl`
- `company_scored.csv`
- `employee_scored.csv`
- `risk_thresholds.json`

## Run (Flask)

```powershell
python app.py
```

Open:

- `http://127.0.0.1:5000`

## Run (Streamlit)

```powershell
streamlit run dashboard.py
```

## First Login

If no admin exists, default admin is auto-created using:

- `ADMIN_USERNAME` (default: `admin`)
- `ADMIN_PASSWORD` (default: `admin123`)

If model files are missing, the app will show an explicit error telling you to run `python fyp.py`.

## Health/Smoke Check

Run local smoke checks:

```powershell
python smoke_check.py
```

This validates key routes and OTP flows.  
Real SMTP checks require network permission in your runtime environment.

## Troubleshooting

- OTP/email not sending:
  - Verify SMTP variables in `.env`
  - For Gmail, confirm App Password and 2-Step Verification
  - Check firewall/network permissions if you see socket access errors
- MongoDB issues:
  - Leave `MONGODB_URI` empty to use CSV fallback mode
- High risk appears too often:
  - Ensure `risk_thresholds.json` exists and is up to date

## License

This project is provided for academic and educational use.
