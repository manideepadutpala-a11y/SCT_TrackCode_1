from flask import Flask, jsonify, send_from_directory
from pathlib import Path
import csv
import traceback
from urllib.parse import quote

from house_prices_linear_regression import train_and_evaluate

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


def load_property_metadata():
    if not TEST_PATH.exists():
        return {}

    with TEST_PATH.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        metadata = {}
        for row in reader:
            property_id = row.get("Id")
            if property_id:
                metadata[property_id] = row
        return metadata


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_display_name(property_id, meta):
    style = meta.get("HouseStyle", "")
    quality = to_int(meta.get("OverallQual", 0))
    area = to_int(meta.get("GrLivArea", 0))

    if quality >= 9 or area >= 2600:
        prefix = "Grand"
    elif quality >= 7 or area >= 1800:
        prefix = "Elegant"
    elif quality >= 5:
        prefix = "Comfort"
    else:
        prefix = "Classic"

    style_names = {
        "2Story": "Two-Story Home",
        "1Story": "Single-Story Home",
        "1.5Fin": "Loft Home",
        "SLvl": "Split-Level Home",
        "SFoyer": "Foyer Home",
    }
    style_name = style_names.get(style, "Family Home")
    return f"{prefix} {style_name} {property_id}"


def classify_property_type(meta):
    style = meta.get("HouseStyle", "")
    bedrooms = to_int(meta.get("BedroomAbvGr", 0))
    quality = to_int(meta.get("OverallQual", 0))
    area = to_int(meta.get("GrLivArea", 0))

    if quality >= 9 or area >= 2600:
        return "Luxury Home"
    if bedrooms >= 4 or area >= 1800:
        return "Family Home"
    if style in {"SLvl", "SFoyer"}:
        return "Split-Level Home"
    if style == "1Story":
        return "Single-Story Home"
    return "Modern Starter Home"


def classify_price_band(price):
    if price >= 380000:
        return "Luxury Range"
    if price >= 280000:
        return "Premium Range"
    if price >= 200000:
        return "Mid Range"
    return "Value Range"


def build_recommendation(item):
    property_type = item["PropertyType"]
    price_band = item["PriceBand"]

    if property_type == "Luxury Home":
        return "Best for buyers looking for premium space and upscale finishes."
    if property_type == "Family Home":
        return "Strong choice for growing families who want more rooms and living area."
    if price_band == "Value Range":
        return "Good entry point for budget-conscious buyers seeking practical space."
    if property_type == "Single-Story Home":
        return "Great for buyers who prefer easy access and a simpler layout."
    return "Balanced option for buyers who want comfort, style, and manageable pricing."


def build_property_insight(item):
    price = item["SalePrice"]
    bedrooms = item["Bedrooms"]
    year_built = item["YearBuilt"]
    area = item["AreaSqFt"]

    if item["PropertyType"] == "Luxury Home":
        return f"Premium pick with {bedrooms} bedrooms, {area} sq ft, and a price near {price:,.0f}."
    if item["PropertyType"] == "Family Home":
        return f"Family-focused layout with {bedrooms} bedrooms and roomy {area} sq ft living space."
    if item["PriceBand"] == "Value Range":
        return f"Budget-friendly option built in {year_built}, ideal for first-time buyers."
    return f"Well-balanced home from {year_built} with {bedrooms} bedrooms and practical everyday comfort."


def build_image_data_uri(item):
    palette = {
        "Luxury Home": ("#6ea8ff", "#2f65ff"),
        "Family Home": ("#67d5b5", "#148f77"),
        "Single-Story Home": ("#ffcf70", "#f39c12"),
        "Split-Level Home": ("#ff9c8c", "#e74c3c"),
        "Modern Starter Home": ("#b39dff", "#7d5fff"),
    }
    start, end = palette.get(item["PropertyType"], ("#7eb6ff", "#3e63dd"))
    badge = item["PriceBand"]
    title = item["DisplayName"]
    subtitle = item["PropertyType"]
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 640 420'>
      <defs>
        <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
          <stop offset='0%' stop-color='{start}' />
          <stop offset='100%' stop-color='{end}' />
        </linearGradient>
      </defs>
      <rect width='640' height='420' rx='32' fill='#0d1623' />
      <rect x='26' y='26' width='588' height='368' rx='28' fill='url(#g)' opacity='0.92' />
      <circle cx='512' cy='102' r='78' fill='rgba(255,255,255,0.16)' />
      <path d='M160 264 L258 180 L356 264 V330 H282 V268 H234 V330 H160 Z' fill='rgba(9,17,29,0.82)' />
      <rect x='386' y='224' width='92' height='106' rx='10' fill='rgba(9,17,29,0.72)' />
      <rect x='404' y='244' width='22' height='22' rx='4' fill='rgba(255,255,255,0.62)' />
      <rect x='438' y='244' width='22' height='22' rx='4' fill='rgba(255,255,255,0.62)' />
      <text x='54' y='74' font-size='30' font-family='Arial, sans-serif' font-weight='700' fill='white'>{badge}</text>
      <text x='54' y='332' font-size='34' font-family='Arial, sans-serif' font-weight='700' fill='white'>{title}</text>
      <text x='54' y='372' font-size='24' font-family='Arial, sans-serif' fill='rgba(255,255,255,0.88)'>{subtitle}</text>
    </svg>
    """.strip()
    return f"data:image/svg+xml;charset=UTF-8,{quote(svg)}"


def build_market_summary(predictions):
    if not predictions:
        return {"highlights": [], "suggestions": []}

    type_seen = set()
    highlights = []
    for item in predictions:
        property_type = item["PropertyType"]
        if property_type in type_seen:
            continue
        type_seen.add(property_type)
        highlights.append({
            "type": property_type,
            "priceBand": item["PriceBand"],
            "message": build_recommendation(item),
        })
        if len(highlights) >= 4:
            break
    suggestions = []
    for item in sorted(predictions, key=lambda row: row["SalePrice"], reverse=True)[:6]:
        suggestions.append({
            "name": item["DisplayName"],
            "type": item["PropertyType"],
            "priceBand": item["PriceBand"],
            "price": round(item["SalePrice"]),
            "image": build_image_data_uri(item),
            "message": build_property_insight(item),
        })
    return {"highlights": highlights, "suggestions": suggestions}


def build_chart_data(predictions):
    property_wave = []
    base_rows = predictions[:12]
    if base_rows:
        values = [row["SalePrice"] for row in base_rows]
        min_value = min(values)
        max_value = max(values)
        value_range = max(max_value - min_value, 1)
        for index, row in enumerate(base_rows):
            normalized = (row["SalePrice"] - min_value) / value_range
            wave_boost = (index % 3 - 1) * 18000 + (index % 2) * 11000
            property_wave.append({
                "label": row["DisplayName"].rsplit(" ", 1)[0],
                "value": max(150000, round(165000 + normalized * 290000 + wave_boost)),
            })

    yearly_buckets = {}
    for row in predictions:
        year = row["YearSold"]
        bucket = yearly_buckets.setdefault(year, {"total": 0.0, "count": 0})
        bucket["total"] += row["SalePrice"]
        bucket["count"] += 1

    yearly_trend = []
    for year in sorted(yearly_buckets):
        bucket = yearly_buckets[year]
        yearly_trend.append({
            "label": str(year),
            "value": round(bucket["total"] / max(bucket["count"], 1)),
        })

    type_totals = {}
    for row in predictions:
        property_type = row["PropertyType"]
        bucket = type_totals.setdefault(property_type, {"total": 0.0, "count": 0})
        bucket["total"] += row["SalePrice"]
        bucket["count"] += 1

    type_comparison = []
    for property_type, bucket in sorted(type_totals.items(), key=lambda item: item[1]["total"] / max(item[1]["count"], 1), reverse=True):
        type_comparison.append({
            "label": property_type,
            "value": round(bucket["total"] / max(bucket["count"], 1)),
        })

    future_projection = []
    if yearly_trend:
        last_year = int(yearly_trend[-1]["label"])
        recent_values = [item["value"] for item in yearly_trend[-3:]]
        if len(recent_values) >= 2:
            changes = [recent_values[idx] - recent_values[idx - 1] for idx in range(1, len(recent_values))]
            avg_change = sum(changes) / len(changes)
        else:
            avg_change = recent_values[0] * 0.04
        current_value = recent_values[-1]
        for step in range(1, 5):
            projected = round(current_value + avg_change * step + current_value * 0.015 * step)
            future_projection.append({
                "label": str(last_year + step),
                "value": projected,
            })

    return {
        "propertyWave": {
            "title": "Property Highlights",
            "subtitle": "A smooth preview of standout homes from the current prediction set.",
            "datasetLabel": "Estimated price",
            "style": "line",
            "points": property_wave,
        },
        "yearlyTrend": {
            "title": "Year-wise Price Trend",
            "subtitle": "Average predicted prices grouped by the sale year in the dataset.",
            "datasetLabel": "Average yearly estimate",
            "style": "bar",
            "points": yearly_trend,
        },
        "futureProjection": {
            "title": "Future Price Projection",
            "subtitle": "A simple forward-looking estimate based on the latest year-wise trend.",
            "datasetLabel": "Projected estimate",
            "style": "line",
            "points": future_projection,
        },
        "typeComparison": {
            "title": "Property Type Comparison",
            "subtitle": "Average pricing by home category to help compare segments.",
            "datasetLabel": "Average type estimate",
            "style": "bar",
            "points": type_comparison,
        },
    }


def load_predictions(limit=20):
    if not SUBMISSION_PATH.exists():
        raise FileNotFoundError("submission.csv not found")

    property_metadata = load_property_metadata()
    with SUBMISSION_PATH.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        predictions = []
        for row in reader:
            if "Id" not in row or "SalePrice" not in row:
                raise ValueError("submission.csv must contain Id and SalePrice columns")
            property_id = row["Id"]
            sale_price = float(row["SalePrice"])
            meta = property_metadata.get(property_id, {})
            prediction = {
                "Id": property_id,
                "DisplayName": build_display_name(property_id, meta),
                "PropertyType": classify_property_type(meta),
                "PriceBand": classify_price_band(sale_price),
                "SalePrice": sale_price,
                "YearSold": to_int(meta.get("YrSold", 0)),
                "YearBuilt": to_int(meta.get("YearBuilt", 0)),
                "Bedrooms": to_int(meta.get("BedroomAbvGr", 0)),
                "AreaSqFt": to_int(meta.get("GrLivArea", 0)),
            }
            prediction["Recommendation"] = build_recommendation(prediction)
            predictions.append(prediction)
            if len(predictions) >= limit:
                break
    return predictions


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/api/predictions")
def api_predictions():
    try:
        predictions = load_predictions(limit=20)
        summary = build_market_summary(predictions)
        return jsonify({
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
            "summary": summary["highlights"],
            "suggestions": summary["suggestions"],
            "charts": build_chart_data(predictions),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 404


@app.route("/api/run", methods=["GET", "POST"])
def api_run_model():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        return jsonify({"success": False, "error": "train.csv or test.csv is missing"}), 400

    try:
        train_and_evaluate(
            str(TRAIN_PATH),
            str(TEST_PATH),
            str(SUBMISSION_PATH),
            model_type="gradient_boosting",
            feature_set="advanced",
            save_model=str(MODEL_PATH),
        )
        predictions = load_predictions(limit=20)
        summary = build_market_summary(predictions)
        return jsonify({
            "success": True,
            "message": "Model run completed",
            "predictions": predictions,
            "count": len(predictions),
            "summary": summary["highlights"],
            "suggestions": summary["suggestions"],
            "charts": build_chart_data(predictions),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
