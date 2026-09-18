# Renewable Energy Analysis

**Can you predict how much solar energy is available from ordinary weather observations?**

Solar output depends on irradiance — how much sunlight actually reaches the panel — and measuring
irradiance needs a pyranometer most sites do not have. Temperature, pressure, humidity, wind speed
and cloud cover, on the other hand, are recorded everywhere by default. If irradiance can be
recovered from those, a site can estimate its solar potential from a weather feed it already has.

This notebook tests that on ~165,000 hourly records, predicting **GHI** (Global Horizontal
Irradiance) from weather features, and compares three models of increasing complexity.

## What it found

| Model | Features | Test R² | Test RMSE |
|---|---|---|---|
| Linear regression | energy delta only | 0.84 | 21.04 |
| XGBoost regressor | 7 weather features | 0.920 | 14.81 |
| Neural network (Keras) | all features | **0.924** | **14.43** |

**Weather alone predicts irradiance well.** The gradient-boosted model reaches R² 0.92 from
temperature, pressure, wind speed, humidity, cloud cover, sunlight-fraction and month — no
irradiance sensor involved.

**The jump from linear to non-linear is where the gain is; the jump from XGBoost to a neural net is
not.** Moving from a straight line to boosted trees buys 0.08 R². Adding a three-layer network on
top of that buys 0.004 more, for a much longer training cycle and a far less interpretable model.
On this data the trees are the better engineering answer, and the notebook shows the comparison
rather than asserting it.

**Train and test scores sit on top of each other** for every model (XGBoost: 0.926 train vs 0.920
test; the network: 0.9239 vs 0.9243), so nothing is straightforwardly overfitting in the usual sense.
See *About the numbers* below for why that agreement deserves a second look.

## How it works

`analysis.ipynb` runs end to end: ETL → exploratory analysis → three models.

**ETL.** Load the hourly dataset, drop the raw time columns in favour of the derived
`SunlightTime/daylength` ratio plus `hour` and `month`, and check nulls and duplicates. There are no
null values anywhere in the frame.

**Exploratory analysis.** A full pairplot across every numerical feature, then an annotated
correlation matrix to find which weather variables actually move with irradiance. That matrix is what
selects the feature set the models use — the choice is shown, not assumed.

**Three models, increasing in capacity.** An ordinary least-squares baseline; an XGBoost regressor
(200 trees, depth 7, learning rate 0.2, subsample and column-sample 0.8, L2 penalty 1000, early
stopping at 10 rounds); and a Keras network — Dense 128 → Dropout → 64 → Dropout → 32 → 1, Adam on
MSE, ten epochs over standardised features.

All five charts are committed with the notebook, so the pairplot, correlation matrix, regression fit
and training-loss curve render on GitHub without running anything.

## Where this goes next

The headline scores look strong, which is exactly why the two most interesting follow-ups are about
how they were measured.

**Deduplicate before splitting.** The dataset carries 69,240 duplicate rows — about 42% of it. The
notebook counts them and keeps them, and the split is a plain random `train_test_split`, so identical
rows land on both sides. Re-running with `df.drop_duplicates()` ahead of the split is the single
highest-value change here: the model *ranking* holds either way, since all three were scored
identically, but the absolute R² values are best read as an upper bound until that rerun happens.

**Split chronologically.** Hourly weather is autocorrelated, and a random split scatters neighbouring
hours across train and test. Training on earlier months and testing on later ones is the test that
matches how the model would actually be used — forecasting forward from a weather feed.

**Then revisit the model choice.** If deduplication compresses the gap between XGBoost and the
network, the trees win outright on training cost and interpretability, and the network can go.

## Data

**The dataset is not included in this repository.** It is
[Renewable Power Generation and Weather Conditions](https://www.kaggle.com/datasets/pythonafroz/renewable-power-generation-and-weather-conditions)
on Kaggle — download it, rename the CSV to `Renewable.csv`, and put it beside the notebook. The
notebook expects one row per hour and these columns:

| Column | Meaning |
|---|---|
| `Energy delta[Wh]` | energy produced in that hour |
| `GHI` | global horizontal irradiance — the prediction target |
| `temp`, `pressure`, `humidity`, `wind_speed` | weather observations |
| `rain_1h`, `snow_1h`, `clouds_all` | precipitation and cloud cover |
| `isSun`, `SunlightTime/daylength` | daylight flags and the sunlight-fraction ratio |
| `weather_type`, `hour`, `month` | categorical weather code and time features |

The notebook also drops `Time`, `sunlightTime` and `dayLength` on load, so a source CSV may include
those. Any hourly solar-and-weather dataset with this shape will run; supply your own and point the
first cell at it.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

Download the Kaggle dataset above and put it beside the notebook as `Renewable.csv` first.

**Keep the scikit-learn pin.** `requirements.txt` pins `scikit-learn<1.6` deliberately: the notebook
calls `mean_squared_error(..., squared=False)`, which was deprecated in 1.4 and **removed in 1.6**, so
the last two model cells raise `TypeError` on a current scikit-learn. Either keep the pin or switch
those calls to `root_mean_squared_error()`.

The neural-network cell trains for ten epochs over ~126,000 rows and takes a few minutes on CPU. The
other cells are quick; the pairplot is the slowest of them.

## Project layout

```
analysis.ipynb      the full analysis - ETL, EDA, three models, all charts committed
requirements.txt    pinned dependencies (see the scikit-learn note above)
```

## Credits

Built with a team of 5.

*Originally built as a course project at Western University.*
