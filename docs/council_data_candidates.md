# Council drainage and culvert data: wave-3 candidates

Drafted 16 Oct 2026 in the council-data thread, for the Stage 2 (ML) thread to choose sources for wave 3.

The draft entries are in `sites/candidates/sources_candidates.yml`, all `enabled: false`. That file is kept apart from `sites/sources.yml` so the two threads never edit the same file. `scripts/probe_candidates.py` runs `probe_source.py --fields --values` on every entry and writes one output file per source to `sites/candidates/probes/`.

The counts below are whole-layer counts. They come from REST metadata and group-by queries made through a web fetcher, so the probe outputs are the check. Some council servers refuse automated fetches (WRC maps, HBRC, Hastings, Whakatāne, Tasman); the extra probes in the script cover those from a normal Python client.

**LiDAR:** every candidate region has 1 m DEM and DSM collections in nz-elevation, including Waikato 2021/2024, Hauraki Plains 2015, BoP 2019-22/2024, Manawatū-Whanganui 2022-24, Otago coastal 2021, Gisborne 2023, Marlborough 2020-22 and Northland 2018-20/2024. `partition.py` (`has_dsm`, `latest_survey`) gives the coverage per tile.

## Ranking against the Stage 2 criteria

| Source (entry) | Region / landscape | Channels | Pipes separated | Culverts | Scope known | Width etc. | Licence |
|---|---|---|---|---|---|---|---|
| `waimakariri_drain_channels_all` | Waimakariri lowlands | ? (probe) | yes (own layer) | (existing) | all drains, council **and private** | topwidth, bottomwidth, depth, lining fields (fill ?) | WDC open data (as now) |
| `waikato_lds_drains` | lower Waikato, Thames Valley/Piako, Franklin, Aka Aka: peat and alluvial flats | 1,470 | yes (drains only) | not public in this layer: WRC asset service (extra probe) | WRC scheme drains only | none | CC BY 4.0 |
| `hauraki_land_drains` | Hauraki Plains | 721 | yes | 3 | HDC district drains only | drain class A/B/C/O/X (meaning to confirm) | CC BY 4.0 |
| `boprc_waterways` + 2 culvert entries | Rangitāiki Plains, Kaituna | 309 | yes | 202 lines + 327 points, diameter on most | scheme/regional drains, owner field | **Width (m) on 244** | "Public available data": confirm |
| `horizons_drainage_channels` + culverts | Manawatū plains (24 schemes) | 765 drains + 34 river channels | yes | 566 floodgate culverts (no size) | Horizons scheme assets | none | not stated: confirm |
| `otago_scheduled_drains` | Taieri Plain, Lower Clutha, Tokomairiro | 317 | yes | none public | ORC scheduled scheme drains | none | CC BY 4.0 |
| `gisborne_rural_drains` | Poverty Bay flats | 283 drains + 41 streams/rivers | yes | none | council public drains, private excluded | none | CC BY 4.0 |
| `ecan_bylaw_drains_2013` / `_2019` | Canterbury, all ECan schemes | 4,888 (2013) / 560 (2019) | yes | none public | ECan-managed drains and watercourses | none; 2019 has SCHEME, CATEGORY | ECan open data |
| `selwyn_land_drains` | Ellesmere lowlands, Leeston | 565 | yes (2 pipes filtered) | — | SDC scheme drains + 120 ECan drains | none | not stated: confirm |
| `selwyn_water_races` / `selwyn_race_culverts` | Selwyn plains | ~2,470 races (race class) | yes | **1,571 race culverts**, diameter on 1,203 | SDC races | — | not stated: confirm |
| `ccc_watercourses` / `ccc_culverts` | Christchurch urban and peri-urban | 9,550 (no class on lines) | yes (separate service) | **2,137**, diameter on 1,573 | CCC-maintained | TopWidth (mm) on 1,239 | CC BY 4.0 |
| `manawatu_dc_open_drains` / `_culverts` | Feilding, rural roads | 278 open + **394 road drains** + 37 swales | yes | 121 | district stormwater | none | confirm |
| `marlborough_drainage_channels` | Wairau Plain | 254 (name only) | yes | — | MEP-scheduled council drains | none | probably CC BY: confirm |
| `marlborough_fish_passage_culverts` | Marlborough, region-wide | — | — | 521 points with barrel sizes (m) | fish-passage survey, all owners | width/height/length | confirm |
| `whakatane_ramm_culverts` | Whakatāne district incl. Rangitāiki Plains | — | — | **4,307 road and side culverts** (points), diameter on most | RAMM road drainage | — | none stated: confirm |
| `hastings_culverts` | Hastings | — | — | 341 | district stormwater | diameter | CC BY 4.0 |
| `waikato_oneview_sw_channels` | 5 Waikato TLAs, mostly urban | ~1,220 | yes | (46 SWDC) | council stormwater | none | CC BY 4.0 |
| `fndc_sw_channels` | Far North towns | 842 (lined/unlined) | yes | not usable (FNDC calls all pipes "Culvert") | town schemes | form | CC BY 4.0 |
| `kaipara_drainage_district_drains` | Northern Wairoa flats (Raupō) | 247 | yes | — | drainage districts | Width on 125, units unknown | disclaimer only: confirm |
| `hurunui_sw_channels` / `_culverts` | Hurunui townships, Amberley | ~570 | yes | 527 | township stormwater | diameter | unknown |
| `ashburton_stockwater` | Ashburton plains | 654 races | yes | — | races, no status (closures) | — | not stated |

## Recommended order for wave 3

1. **New landscapes with clean scheme-drain layers and a clear licence:** `waikato_lds_drains`, `hauraki_land_drains`, `otago_scheduled_drains`, `gisborne_rural_drains`. These are peat, alluvial and coastal flats unlike anything in waves 1 and 2. None has culverts, so they feed the channel model rather than the crossing scorer. All are scheme drains only, so weights like HBRC or Southland's.
2. **Rich but licence to confirm:** `boprc_waterways` (widths, culverts, Rangitāiki Plains) and `horizons_drainage_channels` (Manawatū, the largest scheme set, plus 566 culverts). Both are public views of the councils' own asset registers. An email to each council's GIS team should settle the licence.
3. **Culvert-rich, for the crossing scorer:** `ccc_culverts`, `selwyn_race_culverts`, `whakatane_ramm_culverts`, `marlborough_fish_passage_culverts`. Whakatāne's side culverts are driveway crossings of roadside drains, which the current data lacks. They pair with `boprc_waterways` on the same plains. Selwyn's are crossings of races, not drains: the scorer may need them as their own crossing context.
4. **Probe first, then decide:** `waimakariri_drain_channels_all`. If its widths and private drains are filled, it should replace or extend `waimakariri_channels`.
5. **ECan:** layer 16 (2013) is the full bylaw set (4,888). Layer 3 (2019, the URL of the current disabled `ecan_drains`) has only 560 and looks like the amendment's changes; compare `ASSET_NO` between them. Positions are digitised from plans; the bylaw adds a 10 m buffer for drawing error, which the ±15 m snap corridor should absorb. `TYPE = 'Waterway'` (2,853) is ambiguous between drain and stream, so it is mapped to `other` for now.

**Not recommended:**

- GWRC "Regional highly modified streams" (149k lines): mapped from 2013 DEM flow paths, so circular as labels.
- Horizons One Plan drainage reaches: generalised to a 10 m offset.
- BOPRC river and drain distance centrelines: CC BY-ND.
- ECan's regional "Storm Water" aggregate: pipes and channels are not separated.
- NRC flood protection: stopbanks, floodgates and district polygons only.
- Tasman "LiDAR derived streams": limited access, and circular as labels.
- GWRC flood protection 2016: has 145 drainage channels and 445 culverts, but it is an old SAP snapshot and mostly western river schemes.
- Timaru and Matamata-Piako: small urban sets with no licence stated.

## Open points per source

- **HBRC drains item** (`hbrc_drains`, disabled in sources.yml): it is `AssetManagement/MapServer/3`, so it sits in the same service as the Asset Lines layer (5) behind `hbrc_channels`. Its description says "all known HBRC-managed assets". The extra probes `x_hbrc_drains_item` and `x_hbrc_channels_current` give both sets of counts. The HBRC server blocked the drafting thread's fetcher.
- **WRC culverts, floodgates and pumps:** probably in `WRCMAPS/Flood_protection_and_land_drainage_assets` (`x_wrc_flood_protection_layers` lists its layers); that server refuses web fetchers.
- **Hauraki drain class (Type A/B/C/O/X)** and the DIAMETER codes 1/2/3/6/15: possibly nominal widths. Ask HDC, or check against the DEM after snapping.
- **Swales:** where a council records swales with open drains (Manawatū DC, Hurunui, Selwyn township), the entries follow Auckland's pattern: class `drain` with form `swale`, where a form field exists.
- **Structures for later:** BOPRC point layer (flapgates 255, flood gates 33, pump stations 62); Horizons points (floodgates, pump stations); KDC, FNDC and NRC floodgates (64, 74, 141); Selwyn drain and race points (gates, weirs); CCC WaterCourse service (bridges 8, embankments 10, weirs 11).
- **Stopbanks for later:** BOPRC Asset Register (386), Horizons (314), HDC stopbanks via WRC, ORC floodbanks (bylaw layer 8), KDC, NRC Awanui (1,523).

## Licence follow-ups

These need an email to the council: BOPRC, Horizons, Selwyn, Ashburton, Hurunui, Kaipara, Whakatāne, Manawatū DC, Marlborough.

Training use is internal, but the eventual national product includes all council-recorded crossings, so the licence must cover redistribution of derived data.
