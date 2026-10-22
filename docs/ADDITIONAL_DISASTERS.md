# Additional Disaster Domains — Extension Catalog

Beyond the 9 domains shipped in this extension (oil spills, avalanches, volcanic
eruptions, tsunami inundation, hurricane/cyclone surge, glacial lake outburst
floods, dust storms, wildfire smoke plumes, locust outbreaks), this catalog lists
**21 further disaster types** that can be added with the same adapter pattern.
Each entry maps to one of the existing task contracts (`seg` / `cd` / `scd`) —
no new task handlers are required.

Feasibility tiers:
- **Tier A** — public pixel masks exist; drop-in addition (an adapter + configs + prep README, following `docs/EXTENSION_GUIDE.md`)
- **Tier B** — labels derivable with a prep script; moderate work
- **Tier C** — research-grade only; scarce masks, listed for roadmap completeness

## Tier A — ready to add

| # | Disaster | Task fit | Sensor | Verified public dataset(s) |
|---|----------|----------|--------|----------------------------|
| 10 | Earthquake building damage | `cd` | Optical | **AIDERv2** (Zenodo, 16,723 images, 4 classes); xBD earthquake tier-1 events; CRASAR-U-DROIDs |
| 11 | Tornado / severe wind damage | `cd` | Optical | xBD wind events (documented operational U-Net deployment at Rolling Fork 2023) |
| 12 | Glacier calving fronts | `seg`/`scd` | SAR | **CaFFe** benchmark (681 SAR images, Arctic/Greenland); **CALFIN** (Greenland 1972–2019, Dryad) |
| 13 | Coastal erosion / shoreline retreat | `scd` | Optical | Ian-BD pre/post pairs (DesignSafe, 0.3 m); CoastSat-derived shorelines |
| 14 | Harmful algal blooms | `seg` | Optical | Cyanobacteria concentration rasters (data.gov: Ohio/Florida); DrivenData "Tick Tick Bloom" (Sentinel-2) |
| 15 | Burn severity (upgrades existing wildfire) | `scd` | Optical | CAL FIRE fire perimeters + dNBR pairs; MTBS burn severity |

## Tier B — derivable with prep scripts

| # | Disaster | Task fit | Sensor | Dataset path |
|---|----------|----------|--------|--------------|
| 16 | Tailings dam failure | `cd` | InSAR + optical | Brumadinho ISBAS InSAR time series (Mendeley) + pre/post optical |
| 17 | Deforestation (slow-onset disaster) | `cd` | Optical | PRODES / DETER (INPE, Brazil) |
| 18 | Drought / vegetation die-off | `scd` | Optical | MODIS NDVI anomalies + US Drought Monitor polygons |
| 19 | Urban / industrial fire & explosion | `cd` | Optical | xBD fire events |
| 20 | Sea ice hazard | `seg` | SAR | AI4Arctic Sea Ice (ASIP v2) |
| 21 | Fog episodes (transport hazard) | `seg` | Optical | GOES-R / MODIS fog products |
| 22 | Lahars (volcanic mudflows) | `seg` | Optical | Fuego 2018 / Semeru research mappings |
| 23 | Crop failure / food security | `scd` | Optical | Crop-type time series (CDL; Zimbabwe studies) |

## Tier C — research-grade (roadmap only)

| # | Disaster | Task fit | Blocker |
|---|----------|----------|---------|
| 24 | Sinkholes / subsidence | `seg` | Point inventories only; no pixel masks |
| 25 | River ice jams | `cd` | Sparse, event-based imagery |
| 26 | Permafrost thermokarst | `scd` | Small polygon inventories |
| 27 | Coral bleaching | `seg` | Licensing checks (Allen Coral Atlas) |
| 28 | Iceberg drift hazard | `seg` | USNIC charts need vector→raster pipeline |
| 29 | Conflict damage assessment | `cd` | UNOSAT data; sensitivity/legal review required |
| 30 | Desertification | `scd` | Long-trend Landsat pipelines; heavy prep |

## How to add one

1. Follow `docs/EXTENSION_GUIDE.md`: write an adapter in `src/datasets/` that
   returns the exact tuple for its task contract (channel-first numpy, `255=ignore`).
2. Register it in `src/core/registry.py` (`dataset_libs`, lazy import).
3. Add 2 configs under `configs/<domain>/<dataset>/` (1 lightweight + 1 modern).
4. Add `scripts/data_prep/<domain>/README.md` + `make_dummy.py`; extend
   `scripts/smoke_test_all_domains.py`.
5. Run the smoke test — the whole loop is proven by `make_dummy.py` before any
   real data is downloaded.

**Recommended Phase-2 order:** #10 earthquake → #12 calving fronts → #14 algal
blooms → #11 tornado → #13 coastal erosion (all Tier A; each ≈ one focused session).
