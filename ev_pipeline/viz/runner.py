"""Draw all figures in order (a generator, one yield per figure)."""
import json

import pandas as pd

from .context import build_context
from .figures import (
    activity_overview,
    calibration,
    cluster_geography,
    commitment_costs,
    cumulative_loss,
    error_heatmap,
    feature_gain,
    forecasts,
    holiday_comparison,
    main_accuracy,
    paired_intervals,
    price_schedule,
    window_skill,
    workflow,
    zone_heterogeneity,
)
from .style import STYLE


FIGURES = (
    activity_overview.draw,
    cluster_geography.draw,
    main_accuracy.draw,
    forecasts.draw,
    paired_intervals.draw,
    cumulative_loss.draw,
    window_skill.draw,
    error_heatmap.draw,
    holiday_comparison.draw,
    zone_heterogeneity.draw,
    commitment_costs.draw,
    calibration.draw,
    workflow.draw,
    feature_gain.draw,
    price_schedule.draw,
)


def visualize(data, out, evidence):
    ctx = build_context(data, out, evidence)
    for draw in FIGURES:
        yield from draw(ctx)

    axis_records = ctx.axis_records
    cancellation = ctx.cancellation
    created = ctx.created
    holidays = ctx.holidays
    makeups = ctx.makeups
    nd = ctx.nd
    provenance = ctx.provenance
    res = ctx.res
    text_records = ctx.text_records

    (res / "figure_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    pd.DataFrame(axis_records).to_csv(res / "figure_axes.csv", index=False)
    pd.DataFrame(text_records).to_csv(res / "figure_text.csv", index=False)
    (res / "figure_protocol.json").write_text(
        json.dumps(
            dict(
                width_cm=STYLE["width_cm"],
                dpi=STYLE["dpi"],
                nominal_coverage=0.8,
                bootstrap_block_days=7,
                bootstrap_draws=5000,
                bootstrap_seed=42,
                forecast_origin="23:00",
                validation_days=111,
                common_later_validation_days=81,
                final_window_days=nd - 141,
                training_profile_days=30,
                holidays=holidays,
                make_up_workdays=makeups,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"figures": created, "cancellation": cancellation, "result_dir": res}
