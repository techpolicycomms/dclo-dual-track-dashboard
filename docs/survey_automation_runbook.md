# Survey Automation Runbook

## 1) Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Configure env vars in `.env`:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `TWILIO_FROM_NUMBER`
   - `TWILIO_STATUS_CALLBACK_URL`
   - `SURVEY_PHONE_HASH_SALT`
3. Review `config/survey_automation.yml`.

## 2) Twilio Webhook App (for live calls)

Run locally:

```bash
python src/automation/twilio_webhook_app.py --config config/survey_automation.yml --port 8787
```

Expose publicly (example):

```bash
ngrok http 8787
```

Set `twilio.voice_webhook_url` to `{public_url}/voice/start`.

## 3) One-Command Orchestration

Dry run:

```bash
bash scripts/automation/run_campaign.sh config/survey_automation.yml prototype true
```

Live prototype:

```bash
bash scripts/automation/run_campaign.sh config/survey_automation.yml prototype false
```

## 4) WhatsApp Pilot (Browser Automation)

Generate queue:

```bash
python src/automation/prepare_whatsapp_queue.py --config config/survey_automation.yml --mode cohort1
```

Open links with manual approval:

```bash
python src/automation/run_whatsapp_web_pilot.py --config config/survey_automation.yml
```

## 4A) WhatsApp Business Questionnaire Workflow

Prepare respondent questionnaire state from outreach queue:

```bash
python src/automation/whatsapp_questionnaire_workflow.py --config config/survey_automation.yml --mode prototype --action prepare-state
```

Send the next questionnaire prompt via WhatsApp Business API:

```bash
python src/automation/whatsapp_questionnaire_workflow.py --config config/survey_automation.yml --mode prototype --action send-next --dry-run
```

Run live (without `--dry-run`) after API credentials are set.

## 4B) WhatsApp Mac Demo (to your number)

Preview what will be sent:

```bash
python src/automation/demo_whatsapp_mac.py --phone +41788186778 --question-count 3 --dry-run
```

Send demo messages through WhatsApp for Mac:

```bash
python src/automation/demo_whatsapp_mac.py --phone +41788186778 --question-count 3 --auto-send
```

Notes:
- `--auto-send` uses `osascript` to press Return in WhatsApp after opening each prefilled message.
- macOS may prompt for Accessibility permissions for Terminal/Cursor when using automated keypresses.

## 5) WhatsApp Business API (Scale)

Dry run:

```bash
python src/automation/send_whatsapp_business_api.py --config config/survey_automation.yml --dry-run
```

Live send:

```bash
python src/automation/send_whatsapp_business_api.py --config config/survey_automation.yml
```

## 6) Generated Outputs

- `data/primary/survey_events.jsonl`
- `data/primary/survey_responses.jsonl`
- `data/primary/outreach_queue.csv`
- `data/gold/survey_incentive_eligibility.csv`
- `data/gold/survey_payout_review.csv`
- `data/gold/prototype_ux_report.md`
- `data/gold/dpi_dclo_primary_export.csv`
- `data/gold/dpi_primary_validation_long.csv`

## 7) Live Auto-Sync + Dashboard

Continuously sync incoming survey responses into dashboard-ready files:

```bash
python src/automation/live_data_sync.py --config config/survey_automation.yml --interval-seconds 20
```

Open the live monitoring dashboard:

```bash
streamlit run dashboard/survey_live_dashboard.py --server.port 8502
```

One-command monitoring stack (webhook + tunnel + sync + dashboard):

```bash
bash scripts/automation/run_live_monitoring.sh config/survey_automation.yml 20 8787
```

## 8) Rollback

- Disable live sends:
  - set `twilio.dry_run: true`
  - run WhatsApp API only with `--dry-run`
- Archive current run outputs:
  - move generated files from `data/primary` and `data/gold` into timestamped backup folders.
