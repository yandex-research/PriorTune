#!/usr/bin/env python3
"""Data-aware browser regression checks for PriorTune figures.

Uses the committed research.json as the measurement source. Screenshots require
human visual review; automated checks catch omitted series, fictitious grid
coordinates, inaccessible state, clipping, and data-export regressions.
"""

import argparse
import json
import math
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lgbm", "lgbm-prep", "mlp")
KS = (25, 64, 200)


class Report:
    def __init__(self):
        self.checks = []

    def check(self, condition, name, evidence=None):
        self.checks.append({"pass": bool(condition), "check": name, "evidence": evidence})

    def save(self, output, url):
        passed = sum(row["pass"] for row in self.checks)
        report = {"url": url, "passed": passed, "total": len(self.checks), "checks": self.checks}
        (output / "figure-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{passed}/{len(self.checks)} figure checks passed; report: {output}")
        for row in self.checks:
            if not row["pass"]:
                print("FAIL:", row["check"], row["evidence"])
        return passed == len(self.checks)


def source_checks(data, report):
    """Assert the intended public subset, independently of rendering code."""
    report.check([row["id"] for row in data["models"]] == list(MODELS),
                 "Source model order is LightGBM, LightGBM-TabPrep, MLP")
    for k, side in ((25, 16), (64, 15), (200, 10)):
        for model in MODELS:
            rows = [row for row in data["surfaces"] if row["model"] == model and row["k"] == k]
            ns = sorted({row["n"] for row in rows})
            cs = sorted({row["c"] for row in rows})
            coordinates = {(row["n"], row["c"]) for row in rows}
            report.check(len(rows) == side * side and len(coordinates) == len(rows)
                         and ns == cs and len(ns) == side and min(cs) == k,
                         f"Source grid is complete and starts at K: {model} / {k}",
                         {"cells": len(rows), "coordinates": cs})
            report.check((64 in cs) == (k == 64), f"Only K=64 has a measured coordinate 64: {model} / {k}")
            report.check(all(math.isclose(row["winrate"], .5, abs_tol=1e-12)
                             for row in rows if row["c"] == k),
                         f"Selecting the full pool gives 50%: {model} / {k}")
    report.check(len([row for row in data["elo"] if row["family"] == "tabpack"]) == 2,
                 "Source includes both TabPack reference and selected portfolio")


def finite_svg(svg):
    return svg.evaluate("""svg => [...svg.querySelectorAll('[x],[y],[cx],[cy],[d],[points]')]
      .every(node => [...node.attributes].every(a => !/NaN|Infinity|undefined/.test(a.value)))""")


def text_overflows(svg):
    return svg.evaluate("""svg => {
      const box = svg.getBoundingClientRect();
      return [...svg.querySelectorAll('text')].flatMap(t => {
        const r = t.getBoundingClientRect();
        if (!r.width || !r.height || getComputedStyle(t).display === 'none') return [];
        return r.left < box.left - 2 || r.right > box.right + 2 || r.top < box.top - 2 || r.bottom > box.bottom + 2
          ? [{text:t.textContent, left:r.left-box.left, right:r.right-box.left, top:r.top-box.top, bottom:r.bottom-box.top,
              width:box.width, height:box.height}] : [];
      });
    }""")


def screenshot(page, selector, target):
    page.locator(selector).screenshot(path=str(target),
        style="header,.skip-link,#chart-tooltip { visibility:hidden!important; }")


def panels(page, kind):
    return page.locator(f"#{kind}-chart > [data-model]")


def hp_checks(page, data, report, output):
    rows = data["hyperparameters"]
    defaults = {"lgbm": "learning_rate", "lgbm-prep": "learning_rate", "mlp": "lr"}
    report.check(len(rows) == 36, "All 36 released hyperparameter distributions are available")
    report.check(panels(page, "hp").evaluate_all("nodes=>nodes.map(n=>n.dataset.model)") == list(MODELS),
                 "Three parameter slots use the same model order as the evidence figures")
    for model in MODELS:
        select = page.locator("#hp-key-" + model)
        report.check(select.input_value() == defaults[model], "Parameter slot opens on learning rate: " + model)
        expected = [r for r in rows if r["model"] == model]
        report.check(set(select.locator("option").evaluate_all("nodes=>nodes.map(n=>n.value)")) == {r["key"] for r in expected},
                     "Slot offers every released continuous, categorical and mixed choice: " + model)
        for row in expected:
            select.select_option(row["key"])
            chart = page.locator(f'#hp-{model}-chart')
            slot = page.locator(f'#hp-slot-{model}')
            svg = chart.locator("svg").first
            meta = chart.evaluate("n=>({...n.dataset})")
            distribution = "density" if row["kind"] in ("lin", "log") else "mixed" if row["kind"] == "dropout" else "categorical"
            label = f"{model}/{row['key']}"
            report.check(finite_svg(svg) and meta.get("kind") == row["kind"]
                         and meta.get("distribution") == distribution
                         and int(meta.get("nSelected", -1)) == row["nSelected"],
                         "Rendered distribution type and sample size agree with the source: " + label, meta)
            if row["conditional"]:
                note = slot.locator(".hp-slot-note").inner_text().lower()
                report.check(meta.get("conditional") == "true" and "37" in note and "200" in note
                             and "tabprep" in note and "enabled" in note,
                             "Conditional parameter states its 37-of-200 denominator: " + label, note)
            if distribution == "categorical":
                bars = chart.locator(".hp-bar").evaluate_all("nodes=>nodes.map(n=>({...n.dataset,y:+n.getAttribute('y'),height:+n.getAttribute('height')}))")
                groups = chart.locator(".hp-category")
                valid = len(bars) == len(row["bins"]) * 2 and groups.count() == len(row["bins"])
                geometry = []
                for i, source_bin in enumerate(row["bins"]):
                    pair = chart.locator(f'.hp-category[data-bin-index="{i}"]')
                    actual = pair.locator(".hp-bar").evaluate_all("nodes=>nodes.map(n=>({...n.dataset,y:+n.getAttribute('y'),height:+n.getAttribute('height')}))")
                    lookup = {r.get("series"): r for r in actual}
                    for series in ("prior", "selected"):
                        bar = lookup.get(series, {})
                        valid = valid and math.isclose(float(bar.get("share", "nan")), source_bin[series], abs_tol=1e-12)
                        if series == "selected":
                            valid = valid and int(bar.get("count", -1)) == source_bin["count"]
                        if bar:
                            geometry.append((source_bin[series], bar["height"], bar["y"] + bar["height"]))
                report.check(valid, "Every categorical bar preserves its exact source mass and selected count: " + label, bars)
                scales = [height / share for share, height, _ in geometry if share > 0]
                report.check(bool(scales) and min(scales) > 0 and max(scales) - min(scales) < 1e-7
                             and max(v[2] for v in geometry) - min(v[2] for v in geometry) < 1e-7
                             and all(math.isclose(height, share * (float(meta["plotBottom"]) - float(meta["plotTop"])) / float(meta["yMax"]), abs_tol=1e-7) for share, height, _ in geometry),
                             "Categorical bar heights use a common zero baseline and linear probability scale: " + label)
                report.check(chart.locator(".hp-density-line").count() == 0
                             and "%" in svg.text_content(),
                             "Discrete probabilities use bars and percent units, without smoothing: " + label)
                first = groups.first
                first.focus()
                tip = page.locator("#chart-tooltip")
                source_bin = row["bins"][0]
                report.check(tip.is_visible() and str(source_bin["count"]) in tip.text_content()
                             and str(row["nSelected"]) in tip.text_content(),
                             "Categorical keyboard details show the selected count and denominator: " + label,
                             tip.text_content())
                first.press("ArrowRight")
                report.check(page.evaluate("document.activeElement.dataset.binIndex") == "1"
                             and tip.is_visible(),
                             "Arrow key inspects the next categorical choice: " + label)
                page.keyboard.press("Escape")
                report.check(tip.is_hidden(), "Escape dismisses categorical details: " + label)
                continue
            report.check(chart.locator("path.hp-density-line").count() == 1 and chart.locator(".hp-bar").count() == 0,
                         "Continuous component uses its paper-derived curve: " + label)
            rug_values = [v for v in row["selected"] if distribution != "mixed" or v > 0]
            report.check(chart.locator(".hp-rug").count() == len(rug_values),
                         "Rug preserves the selected continuous values: " + label)
            geometry = chart.evaluate("""node => {
              const prior=node.querySelector('.hp-prior-line'),rug=node.querySelector('.hp-rug');
              return {...node.dataset,d:node.querySelector('.hp-density-line').getAttribute('d'),
                left:+prior.getAttribute('x1'),right:+prior.getAttribute('x2'),one:+prior.getAttribute('y1'),zero:+rug.getAttribute('y1')-4};
            }""")
            numbers = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", geometry["d"])]
            plotted = list(zip(numbers[::2], numbers[1::2]))
            density = row["density"]
            lo, hi = density["domain"]
            correct = len(plotted) == len(density["points"]) == 300
            for point, (px, py) in zip(density["points"], plotted):
                fraction = math.log(point["x"] / lo) / math.log(hi / lo) if density["xScale"] == "log" else (point["x"] - lo) / (hi - lo)
                exact_x = geometry["left"] + fraction * (geometry["right"] - geometry["left"])
                exact_y = geometry["zero"] - point["ratio"] * (geometry["zero"] - geometry["one"])
                correct = correct and math.isclose(px, exact_x, abs_tol=1e-7) and math.isclose(py, exact_y, abs_tol=1e-7)
            report.check(correct and geometry["xScale"] == density["xScale"],
                         "All 300 plotted density samples agree with the paper-derived curve: " + label)
            report.check("relative" in chart.get_attribute("aria-label").lower()
                         and "Density relative to prior" in svg.text_content(),
                         "Density ratio has distinct units from native parameter values: " + label)
            if distribution == "mixed":
                atom = slot.locator(".hp-zero-mass")
                info = atom.evaluate("n=>({...n.dataset})")
                report.check(math.isclose(float(info.get("prior", "nan")), .5)
                             and math.isclose(float(info.get("selected", "nan")), .58)
                             and int(info.get("count", -1)) == 116 and int(info.get("nSelected", -1)) == 200
                             and "116" in atom.inner_text() and "200" in atom.inner_text(),
                             "Dropout shows its 116-of-200 point mass separately from positive density", info)
        select.select_option(defaults[model])
    # Start with unlike distribution types so a shuffle must replace real drawings,
    # including a categorical chart and the mixed dropout distribution.
    for model, key in (("lgbm", "extra_trees"), ("lgbm-prep", "oafe_max_feats"), ("mlp", "dropout")):
        page.locator("#hp-key-" + model).select_option(key)
    screenshot(page, "#inside", output / "1440-hyperparameters-mixed.png")
    for turn in range(4):
        signature = "nodes=>nodes.map(n=>({key:n.dataset.key,d:n.querySelector('svg').innerHTML}))"
        before = panels(page, "hp").evaluate_all(signature)
        page.locator("#hp-shuffle").click()
        page.wait_for_function('!document.getElementById("hp-shuffle").disabled')
        after = panels(page, "hp").evaluate_all(signature)
        report.check(len(after) == 3 and all(a["key"] != b["key"] and a["d"] != b["d"] for a, b in zip(before, after)),
                     f"Lever changes the parameter and drawing in all three slots: pull {turn + 1}")
        report.check(bool(page.locator("#hp-announcement").text_content()), "Shuffle announces the new parameters accessibly")
    with page.expect_download() as event:
        page.locator('[data-download="hyperparameters"]').click()
    target = output / "download-hyperparameters.json"
    event.value.save_as(str(target))
    payload = json.loads(target.read_text())
    report.check(bool(payload.get("provenance")) and len(payload.get("results", [])) == 3
                 and all(row in rows for row in payload["results"]),
                 "HP export preserves the three complete source records and provenance")
    protocol = page.locator("#hp-protocol").text_content().lower()
    report.check("not the probability" in protocol and "categorical" in protocol and "zero" in protocol,
                 "HP disclosure distinguishes densities, discrete choices and the zero atom")
    for model, key in defaults.items():
        page.locator("#hp-key-" + model).select_option(key)


def latency_label_collisions(svg):
    """Use painted bounds, excluding the large invisible point hit targets."""
    return svg.evaluate("""svg => {
      const rect = n => {const r=n.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}};
      const overlap = (a,b) => Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x)>.5 && Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y)>.5;
      const labels=[...svg.querySelectorAll('.ba-direct-label')].map(n=>({
        name:n.textContent,members:JSON.parse(n.dataset.labelMembers||'[]'),box:rect(n.querySelector('rect')||n)}));
      const points=[...svg.querySelectorAll('.ba-mark,.ba-detail-mark')].map(n=>({
        method:n.dataset.method,mode:n.dataset.mode,box:rect(n.querySelector('path'))}));
      const labelPairs=[],coveredMarks=[],measuredOverlaps=[];
      labels.forEach((a,i)=>{labels.slice(i+1).forEach(b=>{if(overlap(a.box,b.box))labelPairs.push([a.name,b.name])});
        points.forEach(p=>{if(!a.members.includes(p.method)&&overlap(a.box,p.box))coveredMarks.push([a.name,p.method,p.mode])})});
      points.forEach((a,i)=>points.slice(i+1).forEach(b=>{if(overlap(a.box,b.box))measuredOverlaps.push([a.method,a.mode,b.method,b.mode])}));
      return {labels:labels.length,labelPairs,coveredMarks,measuredOverlaps};
    }""")


def figure_checks(page, data, report, output):
    hp_checks(page, data, report, output)
    report.check(page.locator("#elo-regime").input_value() == "TE",
                 "BeyondArena opens with T+E deployment")
    report.check("TabPack" in page.locator("#elo-chart").inner_text(),
                 "TabPack is visibly included in the initial BeyondArena comparison")
    for view in ("rank", "latency"):
        page.locator(f'#elo-view [data-view="{view}"]').click()
        for regime in ("TE", "T", "all"):
            page.locator("#elo-regime").select_option(regime)
            marks = page.locator("#elo-chart .ba-mark")
            rendered = marks.evaluate_all("nodes => nodes.map(n => ({...n.dataset}))")
            expected = data["elo" if view == "rank" else "latency"]
            lookup = {(r["method"], r.get("mode", "")): r for r in expected}
            valid, problems = True, []
            axes = page.locator("#elo-chart svg").evaluate("node => ({...node.dataset})")
            misplaced = []
            for row in rendered:
                source = lookup.get((row.get("method"), row.get("mode", "")))
                if source is None or not math.isclose(float(row.get("elo", "nan")), source["elo"], abs_tol=1e-9):
                    valid = False
                    problems.append(row)
                elif view == "latency" and not math.isclose(float(row.get("latencyMs", "nan")), source["latencyMs"], abs_tol=1e-9):
                    valid = False
                    problems.append(row)
                if source is not None:
                    lo, hi = float(axes["xMin"]), float(axes["xMax"])
                    left, right = float(axes["plotLeft"]), float(axes["plotRight"])
                    value = source["elo"] if view == "rank" else source["latencyMs"]
                    fraction = (value - lo) / (hi - lo) if view == "rank" else math.log(value / lo) / math.log(hi / lo)
                    exact_x = left + fraction * (right - left)
                    correct = math.isclose(float(row["x"]), exact_x, abs_tol=1e-7)
                    if view == "latency":
                        low, high = float(axes["yMin"]), float(axes["yMax"])
                        top, bottom = float(axes["plotTop"]), float(axes["plotBottom"])
                        exact_y = bottom - (source["elo"] - low) / (high - low) * (bottom - top)
                        correct = correct and math.isclose(float(row["y"]), exact_y, abs_tol=1e-7)
                    if not correct:
                        misplaced.append(row)
            report.check(bool(rendered) and valid, f"Plotted BeyondArena values match released rows: {view}/{regime}", problems)
            report.check(not misplaced, f"BeyondArena marks occupy their exact measured coordinates: {view}/{regime}", misplaced)
            if regime == "all":
                report.check(len(rendered) == len(expected), f"All published methods are shown: {view}", len(rendered))
            else:
                report.check(all(row.get("regime") in (regime, "D") for row in rendered),
                             f"Deployment filter preserves its regime and labeled defaults: {view}/{regime}")
                report.check(any("TabPack FTE" in row.get("method", "") for row in rendered)
                             and any("TabPack R64" in row.get("method", "") for row in rendered),
                             f"Both TabPack arms remain available: {view}/{regime}")
            report.check(finite_svg(page.locator("#elo-chart svg")), f"Finite SVG geometry: {view}/{regime}")
            if view == "latency":
                collision = latency_label_collisions(page.locator('#elo-chart svg:not([data-secondary="true"])'))
                report.check(not collision["labelPairs"] and not collision["coveredMarks"],
                             "BeyondArena names remain distinct and do not hide unrelated measured marks: " + regime,
                             collision)
                report.check(collision["labels"] == (18 if regime == "all" else 12),
                             "Shared family labels remove repeated selected/random names: " + regime, collision["labels"])
                report.check(page.locator("#elo-chart .ba-pair-link").count() == (0 if regime == "all" else 4),
                             "Default family connectors link exact portfolio/reference points without All-view clutter: " + regime)
    page.locator('#elo-view [data-view="rank"]').click()
    page.locator("#elo-regime").select_option("TE")

    for kind in ("random", "scaling", "heatmap"):
        for k in KS:
            page.locator(f'#{kind}-k [data-k="{k}"]').click()
            group = panels(page, kind)
            order = group.evaluate_all("nodes => nodes.map(n => n.dataset.model)")
            report.check(order == list(MODELS), f"All three models appear in paper order: {kind}/K={k}", order)
            for model in MODELS:
                panel = page.locator(f'#{kind}-chart > [data-model="{model}"]')
                if kind != "heatmap":
                    report.check(panel.locator("svg").count() == 1 and finite_svg(panel.locator("svg")),
                                 f"Model chart renders finite geometry: {kind}/{model}/K={k}")
                    expected_points = (next(r["draws"] for r in data["randomSearch"] if r["model"] == model and r["k"] == k)
                                       if kind == "random" else [r for r in data["budget"] if r["model"] == model and r["k"] == k])
                    report.check(panel.locator(".hit").count() == len(expected_points),
                                 f"Every released curve point has an inspectable marker: {kind}/{model}/K={k}")
                    if kind == "scaling":
                        expected_cells = [r for r in data["budgetCells"] if r["model"] == model and r["k"] == k]
                        report.check(panel.locator(".budget-cell").count() == len(expected_cells),
                                     f"Every evaluated budget appears in the scatter: {model}/K={k}")
                    continue
                rows = [r for r in data["surfaces"] if r["model"] == model and r["k"] == k]
                expected = {(r["n"], r["c"]) for r in rows}
                cells = panel.locator(".heatmap-cell")
                actual = cells.evaluate_all("nodes => nodes.map(n => [Number(n.dataset.n),Number(n.dataset.c)])")
                report.check({tuple(pair) for pair in actual} == expected and len(actual) == len(expected),
                             f"Heatmap has every measured cell and no invented coordinates: {model}/K={k}", len(actual))
                report.check(panel.locator(".heatmap-empty").count() == 0,
                             f"Heatmap contains no phantom empty column: {model}/K={k}")
                report.check(bool(panel.locator(".heatmap-legend, .heatmap-colorbar, .panel-colorbar, .colorbar").count()),
                             f"Panel has its own numeric colour legend: {model}/K={k}")
                scale = panel.evaluate("node => ({...node.dataset})")
                maximum = max(r["winrate"] for r in rows)
                legend_values = panel.locator(".colorbar-values span").all_text_contents()
                report.check(math.isclose(float(scale["colorMin"]), .5)
                             and math.isclose(float(scale["colorMax"]), maximum, abs_tol=1e-12)
                             and float(scale["colorExponent"]) > 1
                             and len(legend_values) == 3
                             and math.isclose(float(legend_values[-1].rstrip("%")) / 100, maximum, abs_tol=.00051),
                             f"Numeric colour legend matches this panel's range: {model}/K={k}", scale)
                colours = cells.evaluate_all("nodes => nodes.map(n => ({n:+n.dataset.n,c:+n.dataset.c,colour:getComputedStyle(n).backgroundColor}))")
                report.check(all(row["colour"] in ("rgb(255, 255, 255)", "rgba(255, 255, 255, 1)")
                                 for row in colours if row["c"] == k),
                             f"Full-pool 50% baseline is white: {model}/K={k}")
                measured = cells.evaluate_all("nodes => nodes.map(n => [Number(n.dataset.n),Number(n.dataset.c),Number(n.dataset.winrate)])")
                lookup = {(r["n"], r["c"]): r["winrate"] for r in rows}
                report.check(all(math.isclose(value, lookup[(n, c)], abs_tol=1e-12) for n, c, value in measured),
                             f"All heatmap measurements match released cells: {model}/K={k}")
                ordered_colours = sorted((lookup[(row["n"], row["c"])],
                                          tuple(map(int, re.findall(r"\d+", row["colour"])[:3]))) for row in colours)
                report.check(ordered_colours[-1][1] == (0, 103, 61)
                             and all(all(a >= b for a, b in zip(left[1], right[1]))
                                     for left, right in zip(ordered_colours, ordered_colours[1:])),
                             f"Colour progression is monotonic and reaches deep green at the panel maximum: {model}/K={k}")
                for pair in (min(expected), max(expected)):
                    n, c = pair
                    cell = panel.locator(f'.heatmap-cell[data-n="{n}"][data-c="{c}"]')
                    cell.click()
                    source = next(r for r in rows if r["n"] == n and r["c"] == c)
                    tip = page.locator("#chart-tooltip")
                    score = tip.locator(".tile-score").text_content().split("%")[0]
                    values = tip.locator(".tile-meta b").all_text_contents()
                    report.check(tip.is_visible() and cell.get_attribute("aria-describedby") == "chart-tooltip"
                                 and math.isclose(float(score) / 100, source["winrate"], abs_tol=.000051)
                                 and [int(v.replace(",", "")) for v in values] == [c, n, c * n],
                                 f"Local tooltip immediately shows the exact selected measurement: {model}/K={k}/{n}/{c}")
                report.check(page.locator(".heatmap-inspector").count() == 0,
                             "Heatmap details do not require a detached bottom inspector")
            with page.expect_download() as event:
                page.locator(f'[data-download="{kind}"]').click()
            target = output / f"download-{kind}-{k}.json"
            event.value.save_as(str(target))
            payload = json.loads(target.read_text())
            rows = payload.get("results", [])
            if kind == "scaling":
                envelope, evaluated = rows.get("envelope", []), rows.get("cells", [])
                report.check(envelope == [r for r in data["budget"] if r["k"] == k]
                             and evaluated == [r for r in data["budgetCells"] if r["k"] == k],
                             f"Scaling export preserves both released envelopes and evaluated budgets: K={k}")
                rows = envelope + evaluated
            report.check(bool(payload.get("provenance")) and {r["model"] for r in rows} == set(MODELS)
                         and {r["k"] for r in rows} == {k},
                         f"Export includes every displayed model and the selected K: {kind}/K={k}")
    protocol = page.locator("#heatmap-protocol").text_content().lower()
    report.check(any(word in protocol for word in ("nonlinear", "non-linear", "power", "gamma", "γ")),
                 "Heatmap discloses its nonlinear colour transformation", protocol)
    report.check("200" in page.locator("#random-protocol").text_content()
                 and "draw" in page.locator("#random-protocol").text_content().lower(),
                 "Random-search band is described as variability across 200 draws")

    for checkbox, kind in (("random-band", "random"), ("scaling-cells", "scaling")):
        checkbox_locator = page.locator("#" + checkbox)
        checkbox_locator.check()
        before = panels(page, kind).evaluate_all("nodes=>nodes.map(n=>n.innerHTML)")
        checkbox_locator.uncheck()
        after = panels(page, kind).evaluate_all("nodes=>nodes.map(n=>n.innerHTML)")
        report.check(len(before) == len(after) == 3 and all(a != b for a, b in zip(before, after)),
                     "Checkbox updates all three panels: " + checkbox)
        checkbox_locator.check()

    for kind, k in (("random", 64), ("scaling", 200), ("heatmap", 25)):
        page.locator(f'#{kind}-k [data-k="{k}"]').click()
    for width in (1440, 1024, 768, 390, 320):
        page.set_viewport_size({"width": width, "height": 1000})
        page.wait_for_timeout(250)
        dimensions = page.evaluate("({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth})")
        report.check(dimensions["scroll"] <= dimensions["width"], f"No horizontal page overflow at {width}px", dimensions)
        for kind in ("random", "scaling", "heatmap", "hp"):
            boxes = panels(page, kind).evaluate_all("nodes=>nodes.map(n=>{let r=n.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})")
            if width == 1440:
                report.check(len(boxes) == 3 and max(b["y"] for b in boxes) - min(b["y"] for b in boxes) < 2
                             and min(b["w"] for b in boxes) >= 250,
                             "Desktop panels align and have usable width: " + kind, boxes)
            if width <= 390:
                report.check(len(boxes) == 3 and boxes[0]["y"] < boxes[1]["y"] < boxes[2]["y"],
                             f"Mobile panels stack in model order: {kind}/{width}px", boxes)
            if kind != "heatmap":
                overflows = [text_overflows(svg) for svg in panels(page, kind).locator("svg").all()]
                report.check(not any(overflows), f"Plot labels stay within SVG: {kind}/{width}px", overflows)
        for selector, name in (("#accuracy", "beyondarena"), ("#random-search", "random"),
                               ("#synthetic", "budget"), ("#scaling-grid", "heatmaps"), ("#inside", "hyperparameters")):
            screenshot(page, selector, output / f"{width}-{name}.png")
        report.check(page.locator("#elo-chart .ba-mark").count() == 16,
                     f"Responsive rank view retains all representative methods at {width}px")
        report.check(not text_overflows(page.locator('#elo-chart svg:not([data-secondary="true"])')),
                     f"Responsive rank labels fit their SVG at {width}px")
        page.locator('#elo-view [data-view="latency"]').click()
        report.check(page.locator("#elo-chart .ba-mark").count() == 26,
                     f"Responsive latency view retains all representative measurements at {width}px")
        report.check(not text_overflows(page.locator('#elo-chart svg:not([data-secondary="true"])')),
                     f"Responsive latency labels fit their SVG at {width}px")
        collision = latency_label_collisions(page.locator('#elo-chart svg:not([data-secondary="true"])'))
        report.check(not collision["labelPairs"] and not collision["coveredMarks"],
                     f"Latency labels avoid one another and unrelated marks at {width}px", collision)
        screenshot(page, "#accuracy", output / f"{width}-latency.png")
        detail = page.locator(".ba-labeled-detail")
        if width <= 390 and detail.count():
            detail.locator("summary").click()
            detail_svg = detail.locator("svg")
            collision = latency_label_collisions(detail_svg)
            report.check(detail_svg.locator(".ba-detail-mark").count() == 26
                         and not collision["labelPairs"] and not collision["coveredMarks"],
                         f"Expanded mobile figure preserves measurements and readable names at {width}px", collision)
            screenshot(page, "#accuracy", output / f"{width}-latency-detail.png")
            detail.locator("summary").click()
        page.locator('#elo-view [data-view="rank"]').click()
        page.evaluate("() => {document.activeElement.blur();window.scrollTo({top:0,behavior:'instant'})}")
        if width in (1440, 390):
            page.screenshot(path=str(output / f"{width}-full.png"), full_page=True)

    page.locator('#heatmap-k [data-k="25"]').click()
    cell = page.locator('.heatmap-cell[data-model="lgbm"][data-n="4000"][data-c="25"]')
    cell.focus()
    report.check(page.locator("#chart-tooltip").is_visible()
                 and "50.00%" in page.locator("#chart-tooltip .tile-score").text_content(),
                 "Keyboard focus exposes the construction baseline immediately")
    cell.press("ArrowRight")
    active = page.evaluate("({...document.activeElement.dataset})")
    report.check(active.get("c") == "50" and active.get("n") == "4000"
                 and page.locator("#chart-tooltip").is_visible(),
                 "Keyboard arrows move to the next measured coordinate and update immediate details")
    page.keyboard.press("Escape")
    report.check(page.locator("#chart-tooltip").is_hidden(), "Escape dismisses the heatmap tooltip")
    page.locator("#elo-chart .ba-mark").first.focus()
    report.check(page.locator("#chart-tooltip").is_visible(), "BeyondArena points expose details on keyboard focus")
    page.keyboard.press("Escape")
    report.check(page.locator("#chart-tooltip").is_hidden(), "Escape dismisses figure details")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:4175/")
    parser.add_argument("--browser")
    parser.add_argument("--output", type=Path, default=ROOT / ".qa" / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = Report()
    data = json.loads((ROOT / "data" / "research.json").read_text())
    source_checks(data, report)
    errors, failed_requests = [], []
    with sync_playwright() as playwright:
        options = {"headless": True}
        chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if args.browser or chrome.is_file():
            options["executable_path"] = args.browser or str(chrome)
        browser = playwright.chromium.launch(**options)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("requestfailed", lambda request: failed_requests.append({"url": request.url, "failure": request.failure}))
        response = page.goto(args.url, wait_until="networkidle")
        page.wait_for_function('document.documentElement.dataset.ready === "true"')
        page.evaluate("window.priortuneDemo.pause()")
        report.check(response.status == 200, "Research page loads successfully")
        figure_checks(page, data, report, args.output)
        report.check(not errors, "No JavaScript errors during all figure interactions", errors)
        report.check(not failed_requests, "No failed figure resource requests", failed_requests)
        browser.close()
    raise SystemExit(0 if report.save(args.output, args.url) else 1)


if __name__ == "__main__":
    main()
