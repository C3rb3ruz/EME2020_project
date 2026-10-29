# app.py
from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd
import os
import sys

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------------------------
for f in ['model.pkl', 'scaler.pkl', 'features.pkl']:
    if not os.path.exists(f):
        sys.exit(f"Missing '{f}'. Run 'python training_model.py' first, from this same folder.")

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# Feature averages (used when a field is left blank) Remove this if we want to entirely disable a feature
FEATURE_MEANS = dict(zip(features, scaler.mean_))

# ---------------------------------------------------------------------------
# Binary feature configuration
# ---------------------------------------------------------------------------
BINARY_FEATURES = [
    'Daytime/evening attendance', 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]
BINARY_LABELS = {f: ('Yes', 'No') for f in BINARY_FEATURES}
BINARY_LABELS['Daytime/evening attendance'] = ('Daytime', 'Evening')
BINARY_LABELS['Gender'] = ('Male', 'Female')

# ---------------------------------------------------------------------------
# Numeric field limits / examples / hints
# ---------------------------------------------------------------------------
FIELD_CONFIG = {
    'Marital status': {
        'min': 1, 'max': 6, 'step': 1,
        'example': '1',
        'hint': '1 = single, 2 = married, 3 = widower, 4 = divorced, 5 = facto union, 6 = legally separated'
    },
    'Application mode': {
        'min': 1, 'max': 57, 'step': 1,
        'example': '1',
        'hint': 'Application type code (e.g. 1 = general contingent)'
    },
    'Application order': {
        'min': 0, 'max': 9, 'step': 1,
        'example': '1',
        'hint': '0 = first choice, 9 = last choice'
    },
    'Course': {
        'min': 33, 'max': 9991, 'step': 1,
        'example': '9500',
        'hint': 'Course code (9500 = Nursing, 9147 = Management)'
    },
    'Previous qualification': {
        'min': 1, 'max': 43, 'step': 1,
        'example': '1',
        'hint': '1 = secondary education'
    },
    'Previous qualification (grade)': {
        'min': 0, 'max': 200, 'step': 'any',
        'example': '133',
        'hint': '0 to 200 scale'
    },
    'Nationality': {
        'min': 1, 'max': 109, 'step': 1,
        'example': '1',
        'hint': '1 = Portuguese'
    },
    'Admission grade': {
        'min': 0, 'max': 200, 'step': 'any',
        'example': '127',
        'hint': '0 to 200 scale'
    },
    'Age at enrollment': {
        'min': 17, 'max': 70, 'step': 1,
        'example': '20',
        'hint': 'Age in years'
    },
    'Unemployment rate': {
        'min': 0, 'max': 30, 'step': 'any',
        'example': '10.8',
        'hint': 'Percent'
    },
    'Inflation rate': {
        'min': -5, 'max': 10, 'step': 'any',
        'example': '1.4',
        'hint': 'Percent, can be negative'
    },
    'GDP': {
        'min': -10, 'max': 10, 'step': 'any',
        'example': '1.74',
        'hint': 'GDP growth, can be negative'
    }
}

for sem in ['1st', '2nd']:
    p = f'Curricular units {sem} sem'
    FIELD_CONFIG.update({
        f'{p} (credited)': {
            'min': 0, 'max': 20, 'step': 1,
            'example': '0',
            'hint': 'Units credited from earlier study'
        },
        f'{p} (enrolled)': {
            'min': 0, 'max': 26, 'step': 1,
            'example': '6',
            'hint': 'Units enrolled in'
        },
        f'{p} (evaluations)': {
            'min': 0, 'max': 45, 'step': 1,
            'example': '8',
            'hint': 'Exams/evaluations taken'
        },
        f'{p} (approved)': {
            'min': 0, 'max': 26, 'step': 1,
            'example': '5',
            'hint': 'Units passed'
        },
        f'{p} (grade)': {
            'min': 0, 'max': 20, 'step': 'any',
            'example': '12.5',
            'hint': 'Average grade, 0-20 scale'
        },
        f'{p} (without evaluations)': {
            'min': 0, 'max': 12, 'step': 1,
            'example': '0',
            'hint': 'Units with no evaluations'
        }
    })

# ---------------------------------------------------------------------------
# Sample-value sidebar & documentation helper
# ---------------------------------------------------------------------------
SAMPLE_VALUE_ROWS = """
<h2>Example values you can copy-paste</h2>
<table style="border-collapse:collapse; width:100%">
<tr style="background:#f0f0f0"><th>Feature</th><th>Sample</th><th>Description</th></tr>
<tr><td>Marital status</td><td>1</td><td>1 = single, 2 = married, 3 = widower, 4 = divorced, 5 = facto union, 6 = legally separated</td></tr>
<tr><td>Application mode</td><td>1</td><td>Code of the admission mode (e.g. 1 = general contingent)</td></tr>
<tr><td>Application order</td><td>1</td><td>0 = first choice, 9 = last choice</td></tr>
<tr><td>Course</td><td>9500</td><td>Course code (9500 = Nursing, 9147 = Management)</td></tr>
<tr><td>Previous qualification</td><td>1</td><td>1 = secondary education</td></tr>
<tr><td>Previous qualification (grade)</td><td>133</td><td>0-200 scale</td></tr>
<tr><td>Nacionality</td><td>1</td><td>1 = Portuguese</td></tr>
<tr><td>Admission grade</td><td>127</td><td>0-200 scale</td></tr>
<tr><td>Age at enrollment</td><td>20</td><td>Age in years</td></tr>
<tr><td>Unemployment rate</td><td>10.8</td><td>Percent</td></tr>
<tr><td>Inflation rate</td><td>1.4</td><td>Percent, can be negative</td></tr>
<tr><td>GDP</td><td>1.74</td><td>GDP growth, can be negative</td></tr>
<tr><td>Curricular units 1st sem (credited)</td><td>0</td><td>Units credited from earlier study</td></tr>
<tr><td>Curricular units 1st sem (enrolled)</td><td>6</td><td>Units enrolled in</td></tr>
<tr><td>Curricular units 1st sem (evaluations)</td><td>8</td><td>Exams/evaluations taken</td></tr>
<tr><td>Curricular units 1st sem (approved)</td><td>5</td><td>Units passed</td></tr>
<tr><td>Curricular units 1st sem (grade)</td><td>12.5</td><td>Average grade 0-20</td></tr>
<tr><td>Curricular units 1st sem (without evaluations)</td><td>0</td><td>Units with no evaluations</td></tr>
<tr><td>Curricular units 2nd sem (credited)</td><td>0</td><td>Units credited from earlier study</td></tr>
<tr><td>Curricular units 2nd sem (enrolled)</td><td>6</td><td>Units enrolled in</td></tr>
<tr><td>Curricular units 2nd sem (evaluations)</td><td>8</td><td>Exams/evaluations taken</td></tr>
<tr><td>Curricular units 2nd sem (approved)</td><td>5</td><td>Units passed</td></tr>
<tr><td>Curricular units 2nd sem (grade)</td><td>12.5</td><td>Average grade 0-20</td></tr>
<tr><td>Curricular units 2nd sem (without evaluations)</td><td>0</td><td>Units with no evaluations</td></tr>
</table>
"""

# ---------------------------------------------------------------------------
# HTML template with Preset Buttons
# ---------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Student Dropout Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 950px; margin: 30px auto; padding: 0 20px; background: #f5f7fa; }
    h1 { color: #0021A5; margin-bottom: 4px; }
    .intro { color: #555; }
    .presets-bar { background: #e8ecef; padding: 12px 16px; border-radius: 6px; margin: 15px 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .presets-bar span { font-weight: bold; font-size: 14px; margin-right: 8px; color: #333; }
    .preset-btn { background: #0021A5; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; }
    .preset-btn:hover { background: #00177a; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 28px; }
    .field { display: flex; flex-direction: column; }
    label { font-size: 14px; font-weight: bold; margin-bottom: 4px; }
    input, select { padding: 7px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; background: white; }
    input:invalid { border-color: #d33; }
    .hint { font-size: 12px; color: #777; margin-top: 3px; }
    button { margin-top: 24px; margin-right: 10px; padding: 10px 24px; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
    .predict { background: #FA4616; color: white; }
    .clear { background: #ddd; }
    #result { margin-top: 20px; padding: 16px; border-radius: 6px; display: none; font-size: 17px; }
    .high { background: #fde2e1; color: #a10000; }
    .low  { background: #e1f7e3; color: #0a6b1a; }
    .err  { background: #fff3cd; color: #7a5b00; }
  </style>
</head>
<body>
  <h1>Student Dropout Predictor</h1>
  <p class="intro">All fields are optional. Blank fields are filled with the training-data average.</p>

  <!-- Quick Test Preset Buttons -->
  <div class="presets-bar">
    <span>Quick Test Profiles:</span>
    <button type="button" class="preset-btn" onclick="loadPreset('low_risk')">Low Risk Student</button>
    <button type="button" class="preset-btn" onclick="loadPreset('high_risk')">High Risk Student</button>
    <button type="button" class="preset-btn" onclick="loadPreset('nursing')">Nursing Student</button>
    <button type="button" class="preset-btn" onclick="loadPreset('struggling')">Struggling Sem</button>
    <button type="button" class="preset-btn" style="background:#555;" onclick="loadPreset('blank')">Clear All</button>
  </div>

  <form id="predictForm">
    <div class="grid">
      {% for feature in features %}
        <div class="field">
          <label for="f{{ loop.index }}">{{ feature }}</label>
          {% if feature in binary_features %}
            <select id="f{{ loop.index }}" data-feature="{{ feature }}">
              <option value="">Not specified</option>
              <option value="1">{{ binary_labels[feature][0] }}</option>
              <option value="0">{{ binary_labels[feature][1] }}</option>
            </select>
          {% else %}
            {% set c = config.get(feature, {}) %}
            <input type="number" id="f{{ loop.index }}" data-feature="{{ feature }}"
                   {% if 'min' in c %}min="{{ c.min }}"{% endif %}
                   {% if 'max' in c %}max="{{ c.max }}"{% endif %}
                   step="{{ c.get('step', 'any') }}"
                   placeholder="e.g. {{ c.get('example', '') }}">
            {% if c.get('hint') or 'min' in c %}
              <span class="hint">{{ c.get('hint', '') }}{% if 'min' in c %} (range {{ c.min }} to {{ c.max }}){% endif %}</span>
            {% endif %}
          {% endif %}
        </div>
      {% endfor %}
    </div>
    <button type="submit" class="predict">Predict</button>
    <button type="reset" class="clear">Clear</button>
  </form>

  <div id="result"></div>

  <hr>

  <!-- Sample-value area rendered safely via Jinja -->
  {{ sample_value_rows | safe }}

  <script>
    // Preset profile data definitions
    const presets = {
      'low_risk': {
        'Marital status': '1',
        'Application mode': '1',
        'Application order': '1',
        'Course': '9500',
        'Previous qualification': '1',
        'Previous qualification (grade)': '140',
        'Nacionality': '1',
        'Admission grade': '135',
        'Daytime/evening attendance': '1',
        'Displaced': '1',
        'Educational special needs': '0',
        'Debtor': '0',
        'Tuition fees up to date': '1',
        'Gender': '0',
        'Scholarship holder': '1',
        'International': '0',
        'Age at enrollment': '19',
        'Unemployment rate': '10.8',
        'Inflation rate': '1.4',
        'GDP': '1.74',
        'Curricular units 1st sem (credited)': '0',
        'Curricular units 1st sem (enrolled)': '6',
        'Curricular units 1st sem (evaluations)': '6',
        'Curricular units 1st sem (approved)': '6',
        'Curricular units 1st sem (grade)': '14.5',
        'Curricular units 1st sem (without evaluations)': '0',
        'Curricular units 2nd sem (credited)': '0',
        'Curricular units 2nd sem (enrolled)': '6',
        'Curricular units 2nd sem (evaluations)': '6',
        'Curricular units 2nd sem (approved)': '6',
        'Curricular units 2nd sem (grade)': '14.0',
        'Curricular units 2nd sem (without evaluations)': '0'
      },
      'high_risk': {
        'Marital status': '1',
        'Application mode': '17',
        'Application order': '3',
        'Course': '9147',
        'Previous qualification': '1',
        'Previous qualification (grade)': '100',
        'Nacionality': '1',
        'Admission grade': '105',
        'Daytime/evening attendance': '1',
        'Displaced': '0',
        'Educational special needs': '0',
        'Debtor': '1',
        'Tuition fees up to date': '0',
        'Gender': '1',
        'Scholarship holder': '0',
        'International': '0',
        'Age at enrollment': '25',
        'Unemployment rate': '13.9',
        'Inflation rate': '2.8',
        'GDP': '-0.92',
        'Curricular units 1st sem (credited)': '0',
        'Curricular units 1st sem (enrolled)': '6',
        'Curricular units 1st sem (evaluations)': '4',
        'Curricular units 1st sem (approved)': '1',
        'Curricular units 1st sem (grade)': '8.0',
        'Curricular units 1st sem (without evaluations)': '2',
        'Curricular units 2nd sem (credited)': '0',
        'Curricular units 2nd sem (enrolled)': '6',
        'Curricular units 2nd sem (evaluations)': '3',
        'Curricular units 2nd sem (approved)': '0',
        'Curricular units 2nd sem (grade)': '0.0',
        'Curricular units 2nd sem (without evaluations)': '3'
      },
      'nursing': {
        'Course': '9500',
        'Tuition fees up to date': '1',
        'Scholarship holder': '1',
        'Age at enrollment': '18',
        'Curricular units 1st sem (enrolled)': '7',
        'Curricular units 1st sem (approved)': '7',
        'Curricular units 1st sem (grade)': '15.0'
      },
      'struggling': {
        'Tuition fees up to date': '1',
        'Debtor': '0',
        'Curricular units 1st sem (enrolled)': '6',
        'Curricular units 1st sem (approved)': '2',
        'Curricular units 1st sem (grade)': '9.5'
      },
      'blank': {}
    };

    function loadPreset(name) {
      const p = presets[name] || {};
      document.querySelectorAll('[data-feature]').forEach(function (el) {
        const feat = el.dataset.feature;
        if (p[feat] !== undefined) {
          el.value = p[feat];
        } else if (name === 'blank') {
          el.value = '';
        }
      });
      document.getElementById('result').style.display = 'none';
    }

    document.getElementById('predictForm').addEventListener('submit', async function (e) {
      e.preventDefault();
      const payload = {};
      document.querySelectorAll('[data-feature]').forEach(function (el) {
        payload[el.dataset.feature] = el.value;
      });
      const box = document.getElementById('result');
      box.style.display = 'block';
      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
          box.className = data.dropout_confidence >= 0.5 ? 'high' : 'low';
          let html = '<strong>' + data.prediction + '</strong><br>Dropout probability: ' + data.confidence_percentage;
          if (data.omitted > 0) {
            html += '<br><small>' + data.omitted + ' field(s) left blank and filled with averages.</small>';
          }
          box.innerHTML = html;
        } else {
          box.className = 'err';
          box.textContent = 'Error: ' + data.error;
        }
      } catch (err) {
        box.className = 'err';
        box.textContent = 'Request failed: ' + err;
      }
    });

    document.getElementById('predictForm').addEventListener('reset', function () {
      document.getElementById('result').style.display = 'none';
    });
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(HTML_PAGE,
                                features=features,
                                binary_features=BINARY_FEATURES,
                                binary_labels=BINARY_LABELS,
                                config=FIELD_CONFIG,
                                sample_value_rows=SAMPLE_VALUE_ROWS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) or {}
        row = {}
        omitted = 0

        # Build the feature vector
        for feature in features:
            val = data.get(feature)

            # Blank -> use training-set average
            if val is None or str(val).strip() == '':
                row[feature] = float(FEATURE_MEANS[feature])
                omitted += 1
                continue

            num = float(val)

            # Validate binary values
            if feature in BINARY_FEATURES:
                if num not in (0, 1):
                    return jsonify(success=False,
                                   error=f"Invalid value for '{feature}': must be 0 or 1."), 400
                row[feature] = num
                continue

            # Validate numeric limits
            cfg = FIELD_CONFIG.get(feature, {})
            if 'min' in cfg and num < cfg['min']:
                return jsonify(success=False,
                               error=f"'{feature}' must be at least {cfg['min']}."), 400
            if 'max' in cfg and num > cfg['max']:
                return jsonify(success=False,
                               error=f"'{feature}' must be at most {cfg['max']}."), 400

            row[feature] = num

        # Scale and predict
        input_df = pd.DataFrame([row], columns=features)
        scaled_input = scaler.transform(input_df)

        probabilities = model.predict_proba(scaled_input)[0]
        dropout_confidence = float(probabilities[1])

        return jsonify({
            'success': True,
            'dropout_confidence': round(dropout_confidence, 4),
            'confidence_percentage': f"{round(dropout_confidence * 100, 2)}%",
            'prediction': 'High Likelihood of Dropout' if dropout_confidence >= 0.5 else 'Low Likelihood of Dropout',
            'omitted': omitted
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ---------------------------------------------------------------------------
# Run the Flask app
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)