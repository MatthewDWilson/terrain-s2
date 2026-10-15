# Council drainage and culvert data: wave-3 candidates

Drafted 16 Oct 2026 in the council-data thread, for the Stage 2 (ML) thread to choose sources for wave 3. Matt ran the first probe the same day: 36 of 38 probes succeeded. This version is revised from those outputs.

The draft entries are in `sites/candidates/sources_candidates.yml` (33 entries, all `enabled: false`). The file is kept apart from `sites/sources.yml` so the two threads never edit the same file. Each entry's comments carry its probe findings, marked "probe 16 Oct".

`scripts/probe_candidates.py` writes one `probe_source.py --fields --values` output per source to `sites/candidates/probes/`, and appends a status block per run to `_summary.txt`.

**LiDAR:** every candidate region has 1 m DEM and DSM collections in nz-elevation. That covers Waikato 2021/2024, Hauraki Plains 2015, BoP 2019-22/2024, Manawatū-Whanganui 2022-24, Otago coastal 2021, Gisborne 2023, Marlborough 2020-22 and Northland 2018-20/2024. `partition.py` (`has_dsm`, `latest_survey`) gives the coverage per tile.

## Rerun needed

The first run had two problems, both now fixed in `probe_candidates.py`:

1. **BOPRC failed.** `boprc_waterways` and `boprc_culverts_line` stopped on a macron (Rangitāiki) printed to the Windows console. Child probes now run in UTF-8.
2. **Value counts covered the whole layer.** `probe_source.py --values` counts the features matching `--where` only, not the entry's `where`, so the counts were over the whole layer rather than the selected features. The runner now passes each entry's `where` as `--where`. The selected counts at the top of each output were already right.

Several entries also changed: some gained or tightened a `where`, and Waikato OneView has a new id field. New entries were added for the WRC flood-protection service (drains, canals, floodgates) and the Waimakariri water races. A full rerun takes about five minutes:

```
python scripts/probe_candidates.py
```

## Results against the Stage 2 criteria

| Source (entry) | Region / landscape | Channels (probed) | Pipes separated | Culverts | Scope | Width etc. | Licence |
|---|---|---|---|---|---|---|---|
| `waikato_lds_drains` | lower Waikato, Thames Valley/Piako, Franklin, Aka Aka: peat and alluvial flats | 1,470 | yes (drains only) | see `wrc_floodgates` | WRC scheme drains only | none | CC BY 4.0 |
| `wrc_fp_drains` / `wrc_fp_canals` / `wrc_floodgates` | same | not yet probed | yes (own layers) | floodgates (count ?) | WRC flood protection and drainage | ? | WRC: check |
| `hauraki_land_drains` | Hauraki Plains | 721 | yes | 3 | HDC district drains only | drain class A/B/C/O/X (meaning to confirm) | CC BY 4.0 |
| `boprc_waterways` + 2 culvert entries | Rangitāiki Plains, Kaituna | 309 | yes | 202 lines + 327 points; DiameterMM on most points (median 530) | scheme/regional drains; owner field | **Width (m) on 244** | "Public available data": confirm |
| `horizons_drainage_channels` + culverts | Manawatū plains (24 schemes) | 800 (765 drains, 34 river channels, 1 toe drain) | yes | 566 floodgate culverts (no size) | Horizons scheme assets (owner Horizons 99 %) | none | not stated: confirm |
| `otago_scheduled_drains` | Taieri Plain, Lower Clutha, Tokomairiro | 317 | yes | none public | ORC scheduled scheme drains | none | CC BY 4.0 |
| `gisborne_rural_drains` | Poverty Bay flats | 283 drains + 41 streams/rivers + 1 diversion | yes | none | council public drains; private excluded | none | CC BY 4.0 |
| `ecan_bylaw_drains_2019` | Canterbury, all ECan drainage schemes | 559 (436 drains, 73 modified, 47 natural, 4 grassed) | yes | none public | ECan scheduled drains; SCHEME, CATEGORY A/B/C, CLEANING | none | ECan open data |
| `ecan_bylaw_drains_2013` | Canterbury | 4,888 (2,853 "Waterway", 1,598 natural, 432 drains) | yes | — | ECan 2013 schedule | none | ECan open data |
| `selwyn_land_drains` | Ellesmere lowlands, Leeston | 563 | yes (2 pipes filtered) | — | SDC scheme drains + 120 ECan drains; mostly from old drain maps | none | not stated: confirm |
| `selwyn_water_races` / `selwyn_race_culverts` | Selwyn plains | 2,454 races (race class) | yes | **1,571 race culverts** (diameter, mm, on some) | SDC races; Private 285 | — | not stated: confirm |
| `waimakariri_water_races` | Waimakariri plains | ~600 races (race class) | yes | — | farmer- and WIL-maintained races | — | WDC open data |
| `ccc_watercourses` / `ccc_culverts` | Christchurch urban and peri-urban | 9,550 (no class on lines) | yes (separate service) | **2,137** (before the In Service filter), diameter on most | CCC-maintained; 3.5 % private | TopWidth (mm) on about 28 % | CC BY 4.0 |
| `manawatu_dc_open_drains` / `_culverts` | Feilding, Halcombe, Sanson, Oroua Downs, Rongotea | 709 (**394 road drains**, 278 open, 37 swales) | yes | 121 (diameter mm) | road, utility **and private (121)** drains | channel class 1-4 | confirm |
| `marlborough_drainage_channels` | Wairau Plain | 254 (name only) | yes | — | MEP-scheduled council drains | none | probably CC BY: confirm |
| `marlborough_fish_passage_culverts` | Marlborough, region-wide | — | — | 521 points | fish-passage survey, all owners | sizes on ~12 only | confirm |
| `whakatane_ramm_culverts` | Whakatāne district incl. Rangitāiki Plains | — | — | **4,307 road and side culverts** (points), diameter on about two-thirds | RAMM road drainage | — | none stated: confirm |
| `hastings_culverts` | Hastings | — | — | 341 (diameter mm) | district stormwater | — | CC BY 4.0 |
| `waikato_oneview_sw_channels` | 5 Waikato TLAs, mostly urban | 1,222 | yes | — | council stormwater | none | CC BY 4.0 |
| `fndc_sw_channels` | Far North towns | 842 (lined/unlined) | yes | not usable (FNDC calls all pipes "Culvert") | town schemes; now In service only | form | CC BY 4.0 |
| `kaipara_drainage_district_drains` | Northern Wairoa flats (Raupō) | 247 | yes | — | drainage districts | **Width and Depth on all; units probably feet** | disclaimer only: confirm |
| `hurunui_sw_channels` / `_culverts` | Amberley, Hanmer Springs, Cheviot (townships) | 568 | yes | 527 | township stormwater; owner fields empty | diameter | unknown |
| `ashburton_stockwater` | Ashburton plains | 654 races | yes | — | races, no status (closures) | — | not stated |
| `waimakariri_drain_channels_all` | Waimakariri rural schemes | 637 (2012/13 snapshot) | yes | — | WDC scheme channels; wet/dry flag | **all width fields empty** | WDC open data |

## What the probe changed

- **Waimakariri "all drain channels" (layer 5) is not an upgrade.**
  - Despite the layer description, it holds 637 council-owned lines from 2012/13, and every width, depth and lining field is empty.
  - It is smaller and older than `waimakariri_channels`, which has 3,174 lines in the wave 1+2 tiles alone. Its one extra is a wet/dry flag (Wet Channel 346, Dry Channel 214).
  - The same service has a Water Race layer: 651 lines, 364 of them farmer-maintained. It is now a candidate (`waimakariri_water_races`, race class).
- **HBRC drains item (`hbrc_drains`, disabled in sources.yml) adds nothing.** Its layer (AssetManagement/3) holds 702 lines: 572 Drain and 130 Pipeline. These match the Drain and Pipeline rows of the Asset Lines layer that `hbrc_channels` already reads. Leave it disabled. If anything is ever taken from it, `Pipeline` must be excluded.
  - The same service lists Pump Stations (layer 8) and Asset Points (layer 4), which are structures candidates.
- **ECan: use the 2019 layer (3) for drains and the 2013 layer (16) for waterways only.**
  - The 2019 layer has 436 drains against 432 in 2013, plus SCHEME, CATEGORY and CLEANING. It looks like the current drain schedule, with the river-scheme waterways removed.
  - One 2019 line is a 3,300 km artefact (LENGTH_M 3,339,652). The entry now filters it out with `LENGTH_M < 50000`.
  - The 1,598 "Natural waterway" rows in 2013 have no digitiser, so they were probably copied from topographic data.
  - The current disabled `ecan_drains` entry in sources.yml points at layer 3, which is the right choice for drains, but it needs this filter.
- **WRC flood-protection service answers Python.** It blocks web fetchers but not the probe. Its layers are Pump station (127), Floodgate, Control gate, Embankments, Drain, Canal and Drainage Scheme. The three new entries (`wrc_fp_drains`, `wrc_fp_canals`, `wrc_floodgates`) still need their first probe. `wrc_fp_drains` may duplicate `waikato_lds_drains`.
- **Kaipara widths look like feet.** Width and Depth are filled on all 247 lines: Width median 4, max 24; Depth median 3, max 7. Depths of 3-7 m are implausible for farm drains, whereas 6 × 4 ft (1.8 × 1.2 m) is a typical drain section. The entry maps width with 0.3048; check this against the snapped DEM widths.
- **Manawatū DC open drains are broader than expected.** They include 121 private drains and a 1-4 channel class, and 119 lines have "Very Poor" accuracy. They are mostly roadside and village drains on flat land, which the crossing scorer lacks elsewhere.
- **Marlborough fish-passage culverts have almost no sizes** (about 12 points). Records carry REC segment ids, so positions may be snapped to REC lines; check offsets before relying on them. Its 699 bridges could feed the bridge class.
- **Fixes to individual entries:**
  - Waikato OneView: `AssetID` is empty on every row, so the id is now `OneviewID`. The layer extent is corrupt, so always query it with a window.
  - CCC culverts: about 11 % of rows are abandoned or removed, so the entry now keeps `ServiceStatus = 'In Service'`.
  - Hastings culverts: the entry now keeps `IPS_Service_Status = 'INS'`.
  - FNDC channels: the entry now keeps `service_status = 'In service'`.
  - Selwyn race culverts: the entry now keeps `STATUS = 'In Service'`.
- **Searches** for Tasman, Tararua, Greater Wellington drainage-scheme drains and Hastings open drains found nothing new on ArcGIS Online.

## Recommended order for wave 3

1. **New landscapes with clean scheme-drain layers and a clear licence:** `waikato_lds_drains`, `hauraki_land_drains`, `otago_scheduled_drains`, `gisborne_rural_drains`. These are peat, alluvial and coastal flats unlike anything in waves 1 and 2.
   - All four probes are clean and every mapped field is present.
   - None has culverts, so they feed the channel model rather than the crossing scorer, unless `wrc_floodgates` probes well.
   - All are scheme drains only, so weights like HBRC or Southland's.
2. **Rich but licence to confirm:** `boprc_waterways` (widths, culverts, Rangitāiki Plains; rerun pending) and `horizons_drainage_channels` (Manawatū, 800 channels and 566 culverts; probe clean). Both are public views of the councils' own asset registers; an email to each council's GIS team should settle the licence.
3. **Culvert-rich, for the crossing scorer:**
   - `ccc_culverts`.
   - `selwyn_race_culverts`: crossings of races, not drains, so the scorer may need them as their own context.
   - `whakatane_ramm_culverts`: 1,536 side culverts are driveway crossings of roadside drains, which the current data lacks. They pair with `boprc_waterways` on the same plains.
   - `manawatu_dc_culverts` with `manawatu_dc_open_drains`: road drains and their culverts together.
4. **ECan:** take `ecan_bylaw_drains_2019` (or fix `ecan_drains` in sources.yml with the same filter). Weigh the 2013 waterways separately.
5. **Races:** `selwyn_water_races`, `waimakariri_water_races` and `ashburton_stockwater` give the race class real examples; Ashburton's have no status, and closures are under way.

**Not recommended:**

- `waimakariri_drain_channels_all`: an old subset with empty widths.
- `hbrc_drains`: duplicates Asset Lines.
- GWRC "Regional highly modified streams" (149k lines): mapped from 2013 DEM flow paths, so circular as labels.
- Horizons One Plan drainage reaches: generalised to a 10 m offset.
- BOPRC river and drain distance centrelines: CC BY-ND.
- ECan's regional "Storm Water" aggregate: pipes and channels are not separated.
- NRC flood protection: stopbanks, floodgates and district polygons only.
- Tasman "LiDAR derived streams": limited access, and circular as labels.
- GWRC flood protection 2016: old SAP snapshot, mostly western river schemes.
- Timaru and Matamata-Piako: small urban sets with no licence stated.

## Open points

- Hauraki drain class (Type A/B/C/O/X) and the DIAMETER codes 1/2/3/6/15: possibly nominal widths. Ask HDC, or compare with DEM widths after snapping.
- Positional quality varies:
  - Selwyn land drains: mostly digitised from old drain maps.
  - ECan: digitised from plans; the bylaw adds a 10 m buffer for drawing error.
  - CCC culverts: 43 % "Non Verified".
  - Manawatū DC: 119 lines "Very Poor".
  - The ±15 m snap corridor should absorb most of this. Unsnapped lines already get weight 0.5.
- Swales: Manawatū DC and Hurunui record swales with their open drains. Manawatū's map to form `swale` (Auckland's pattern). Hurunui has no form field, so its swales are mapped as drains: drop them from the `where` if ML-3b sets swale weight to 0.
- **Structures for later:**
  - BOPRC point layer: flapgates 255, flood gates 33, pump stations 62.
  - Horizons points: floodgates, pump stations.
  - WRC: pump stations, floodgates, control gates.
  - HBRC: pump stations.
  - KDC, FNDC and NRC floodgates (64, 74, 141).
  - Selwyn drain and race points (gates, weirs).
  - CCC WaterCourse service (bridges, embankments, weirs).
- **Stopbanks for later:** BOPRC Asset Register (386), Horizons (314), WRC Embankments, HDC stopbanks via WRC, ORC floodbanks (bylaw layer 8), KDC, NRC Awanui (1,523).

## Licence follow-ups

These need an email to the council: BOPRC, Horizons, Selwyn, Ashburton, Hurunui, Kaipara, Whakatāne, Manawatū DC, Marlborough.

Training use is internal, but the eventual national product includes all council-recorded crossings, so the licence must cover redistribution of derived data.
