import csv
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

START_YEAR = 1895
START_MONTH = 6
MONTHS = 127  # 1895-06 through 1905-12


def month_date(index):
    month0 = START_MONTH - 1 + index
    return date(START_YEAR + month0 // 12, month0 % 12 + 1, 1)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def simulate(terrain_multiplier=1.0, urban_multiplier=1.0, governance_speed=1.0):
    dates = [month_date(i) for i in range(MONTHS)]
    control = np.zeros(MONTHS)
    resistance = np.zeros(MONTHS)
    governance = np.zeros(MONTHS)
    services = np.zeros(MONTHS)
    legitimacy = np.zeros(MONTHS)
    disease_pressure = np.zeros(MONTHS)
    fiscal_expense = np.zeros(MONTHS)
    fiscal_revenue = np.zeros(MONTHS)

    control[0], resistance[0], governance[0] = 0.12, 0.88, 0.05
    services[0], legitimacy[0], disease_pressure[0] = 0.08, 0.04, 1.0

    for t in range(MONTHS - 1):
        d = dates[t]
        campaign = 1.0 if d <= date(1895, 11, 1) else 0.35
        reform = sigmoid((t - 48) / 11.0)  # fiscal effects lag institutional reform
        public_works_burden = 1.40 * math.exp(-((t - 72) / 30.0) ** 2)
        terrain = terrain_multiplier * (0.58 + 0.22 * resistance[t])
        urban = urban_multiplier * (0.28 + 0.20 * (1.0 - services[t]))

        control_gain = 0.24 * campaign * (1.0 - control[t])
        control_loss = 0.015 * resistance[t] * (terrain + urban) * control[t]
        control[t+1] = np.clip(control[t] + control_gain - control_loss, 0.0, 1.0)

        grievance = 0.012 * (1.0 - legitimacy[t]) + 0.008 * urban
        demobilization = (0.020 * control[t] + 0.080 * control[t] * governance[t]
                          + 0.025 * services[t])
        resistance[t+1] = np.clip(
            resistance[t] + grievance * (1.0 - resistance[t]) - demobilization * resistance[t],
            0.0, 1.0)

        governance_gain = governance_speed * (0.010 + 0.018 * reform) * control[t]
        governance_loss = 0.004 * resistance[t] * terrain
        governance[t+1] = np.clip(governance[t] + governance_gain - governance_loss, 0.0, 1.0)

        service_gain = governance_speed * (0.010 + 0.016 * reform) * governance[t]
        service_loss = 0.006 * resistance[t] * urban
        services[t+1] = np.clip(services[t] + service_gain - service_loss, 0.0, 1.0)

        coercion = 0.10 * campaign + 0.035 * resistance[t]
        legitimacy[t+1] = np.clip(
            legitimacy[t] + 0.016 * services[t] + 0.010 * governance[t]
            - coercion * (1.0 - legitimacy[t]), 0.0, 1.0)

        disease_pressure[t+1] = max(
            0.08, disease_pressure[t] * (0.94 - 0.025 * services[t])
            + 0.035 * campaign * terrain)

        # Million-yen annual-rate equivalents. Security, health and infrastructure
        # initially dominate; revenue rises through land/fiscal reform and monopolies.
        fiscal_expense[t] = (5.0 + 2.8 * resistance[t] + 2.2 * disease_pressure[t]
                             + 1.6 * (1.0 - governance[t]) + 5.2 * reform
                             + public_works_burden)
        fiscal_revenue[t] = (1.65 + 1.7 * governance[t] + 2.2 * reform
                             + 1.5 * control[t] * governance[t])

    fiscal_expense[-1] = (5.0 + 2.8 * resistance[-1] + 2.2 * disease_pressure[-1]
                          + 1.6 * (1.0 - governance[-1]) + 5.2
                          + 1.40 * math.exp(-((MONTHS - 1 - 72) / 30.0) ** 2))
    fiscal_revenue[-1] = (1.65 + 1.7 * governance[-1] + 2.2
                          + 1.5 * control[-1] * governance[-1])

    # Anchor annual fiscal gap to the documented roughly 4 million yen in FY1896.
    mask_1896 = np.array([d.year == 1896 for d in dates])
    raw_gap_1896 = np.mean(fiscal_expense[mask_1896] - fiscal_revenue[mask_1896])
    fiscal_revenue += raw_gap_1896 - 4.0

    # Campaign medical burden is distributed by exposure and normalized to sources.
    campaign_mask = np.array([d <= date(1896, 5, 1) for d in dates])
    exposure = disease_pressure * campaign_mask
    disease_cases = exposure / exposure.sum() * 26_094
    disease_deaths = exposure / exposure.sum() * 4_642
    battle_weight = np.exp(-np.arange(MONTHS) / 2.0) * (np.arange(MONTHS) < 6)
    battle_deaths = battle_weight / battle_weight.sum() * 164
    battle_wounded = battle_weight / battle_weight.sum() * 515

    occupation_cost = (fiscal_expense + 0.00012 * disease_cases
                       + 1.8 * resistance * terrain_multiplier
                       + 0.9 * urban_multiplier * (1.0 - services))
    return {
        "dates": dates,
        "control": control,
        "resistance": resistance,
        "governance": governance,
        "services": services,
        "legitimacy": legitimacy,
        "disease_pressure": disease_pressure,
        "fiscal_expense": fiscal_expense,
        "fiscal_revenue": fiscal_revenue,
        "disease_cases": disease_cases,
        "disease_deaths": disease_deaths,
        "battle_deaths": battle_deaths,
        "battle_wounded": battle_wounded,
        "occupation_cost": occupation_cost,
    }


def annualize(sim):
    years = sorted(set(d.year for d in sim["dates"]))
    rows = []
    for year in years:
        idx = np.array([d.year == year for d in sim["dates"]])
        rows.append({
            "year": year,
            "control": float(np.mean(sim["control"][idx])),
            "resistance": float(np.mean(sim["resistance"][idx])),
            "governance": float(np.mean(sim["governance"][idx])),
            "services": float(np.mean(sim["services"][idx])),
            "legitimacy": float(np.mean(sim["legitimacy"][idx])),
            "expense_million_yen": float(np.mean(sim["fiscal_expense"][idx])),
            "revenue_million_yen": float(np.mean(sim["fiscal_revenue"][idx])),
            "fiscal_gap_million_yen": float(np.mean(sim["fiscal_expense"][idx]
                                                      - sim["fiscal_revenue"][idx])),
            "occupation_cost_index": float(np.mean(sim["occupation_cost"][idx])),
        })
    return rows


def font(size, bold=False):
    p = Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc")
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()


def chart(path, title, years, series, y_max=1.0, note=""):
    w, h = 1500, 850
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 145, 105, 70, 145
    pw, ph = w-left-right, h-top-bottom
    draw.text((left, 30), title, fill="#111827", font=font(34, True))
    for j in range(6):
        y = top + ph*j/5
        value = y_max*(1-j/5)
        draw.line((left,y,left+pw,y),fill="#e5e7eb",width=2)
        draw.text((35,y-13),f"{value:.2f}",fill="#4b5563",font=font(19))
    for label, values, color in series:
        pts=[]
        for i,v in enumerate(values):
            x=left+pw*i/max(1,len(values)-1)
            y=top+ph*(1-min(v,y_max)/y_max)
            pts.append((x,y))
        draw.line(pts,fill=color,width=5)
    for yr in (1895,1898,1901,1905):
        i=years.index(yr)
        x=left+pw*i/max(1,len(years)-1)
        draw.text((x-25,top+ph+15),str(yr),fill="#374151",font=font(19))
    x=left
    for label,_,color in series:
        draw.line((x,h-70,x+38,h-70),fill=color,width=6)
        draw.text((x+48,h-84),label,fill="#111827",font=font(19))
        x+=320
    draw.text((left,h-118),note,fill="#6b7280",font=font(17))
    img.save(path)


def main():
    base = simulate()
    annual = annualize(base)
    years = [r["year"] for r in annual]

    chart(OUT / "乙未战争_控制抵抗治理状态.png",
          "Yiwei War aftermath: control, resistance, and governance",
          years,
          [("Military control",[r["control"] for r in annual],"#1d4ed8"),
           ("Residual resistance",[r["resistance"] for r in annual],"#b91c1c"),
           ("Governance capacity",[r["governance"] for r in annual],"#166534"),
           ("Public services",[r["services"] for r in annual],"#7c3aed")],
          note="Latent indices are model states, not direct historical measurements.")

    gaps = [r["fiscal_gap_million_yen"] for r in annual]
    ymax=max(max(gaps),4.2)
    chart(OUT / "乙未战争_殖民财政缺口.png",
          "Modeled colonial fiscal gap after 1895",
          years,
          [("Fiscal gap (million yen)",gaps,"#a16207")],
          y_max=ymax,
          note="FY1896 is anchored near 4 million yen; later path is a model reconstruction.")

    scenarios=[]
    for terrain in (0.8,1.0,1.2,1.4):
        for urban in (0.8,1.0,1.2,1.4):
            for gov in (0.8,1.0,1.2):
                sim=simulate(terrain,urban,gov)
                ann=annualize(sim)
                first_self=next((r["year"] for r in ann if r["fiscal_gap_million_yen"]<=0),None)
                scenarios.append({
                    "terrain_multiplier":terrain,"urban_multiplier":urban,
                    "governance_speed":gov,"first_fiscal_self_reliance_year":first_self,
                    "resistance_1902":next(r["resistance"] for r in ann if r["year"]==1902),
                    "cumulative_cost_index":float(np.sum(sim["occupation_cost"])),
                })

    with (OUT / "乙未战争_年度状态与占领成本.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=annual[0].keys())
        writer.writeheader(); writer.writerows(annual)
    with (OUT / "乙未战争_地形治理敏感性.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=scenarios[0].keys())
        writer.writeheader(); writer.writerows(scenarios)

    summary={
        "historical_calibration_targets":{
            "battle_deaths":164,"battle_wounded":515,
            "disease_cases":26094,"disease_deaths":4642,
            "fiscal_gap_1896_million_yen":4.0,
            "fiscal_self_reliance_reference":"approximately 1904-1905",
        },
        "model_output":{
            "battle_deaths":float(base["battle_deaths"].sum()),
            "battle_wounded":float(base["battle_wounded"].sum()),
            "disease_cases":float(base["disease_cases"].sum()),
            "disease_deaths":float(base["disease_deaths"].sum()),
            "disease_to_battle_death_ratio":float(base["disease_deaths"].sum()/base["battle_deaths"].sum()),
            "fiscal_gap_1896_million_yen":next(r["fiscal_gap_million_yen"] for r in annual if r["year"]==1896),
            "first_fiscal_self_reliance_year":next((r["year"] for r in annual if r["fiscal_gap_million_yen"]<=0),None),
            "resistance_index_1902":next(r["resistance"] for r in annual if r["year"]==1902),
        },
        "sensitivity_scenario_count":len(scenarios),
        "warning":"Latent control/resistance/governance indices are explanatory constructs, not observed historical series.",
    }
    (OUT/"乙未战争_占领成本模型摘要.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
